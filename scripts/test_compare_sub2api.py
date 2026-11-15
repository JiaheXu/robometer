import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from PIL import Image

spec=importlib.util.spec_from_file_location('comparison',Path(__file__).with_name('compare_sub2api.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.head=self.root/'head';self.three=self.root/'three';self.source=self.root/'source';self.out=self.root/'out'
        for p in (self.head,self.three,self.source):p.mkdir()
        Image.new('RGB',(1280,720),'green').save(self.source/'head.png')
        Image.new('RGB',(1281,240),'blue').save(self.three/'tile.png')
        self.job=dict(episode='episode1',event_id='0000',segment=0,seconds=0.,source_timestamp_ns=100,
            next_subtask_instruction='Put the cup on the shelf.',context_event_ids=['0000'],
            images={'cam_head':'head.png'},head_image_sha256=c.sha((self.source/'head.png').read_bytes()),
            composite_image='tile.png',composite_sha256=c.sha((self.three/'tile.png').read_bytes()),
            camera_provenance={camera:{'source_timestamp_ns':100} for camera in ('head','left','right')})
        for p in (self.head,self.three):
            (p/'manifest.json').write_text(json.dumps({'source':str(self.source),'jobs':[self.job]}))
            (p/'results.jsonl').write_text(json.dumps({**self.job,'progress':.99,'success_probability':.98})+'\n')
        self.scores=dict(progress=.4,success_probability=.2,status='not_success',evidence='Cup is still held.')

    def body(self,mode):
        return c.request_body(self.job,mode,{('episode1','0000'):self.job},self.source,self.three,'gpt-6-astra','medium')

    def test_both_inputs_are_blind_and_use_original_dimensions(self):
        self.job.update(progress=.99,success_probability=.98,review_notes=['Fake success label'])
        for mode,size in [('head',[427,240]),('three',[1281,240])]:
            body,images=self.body(mode)
            self.assertFalse(body['store']);self.assertEqual(body['model'],'gpt-6-astra')
            self.assertEqual(images[0]['dimensions'],size)
            content=body['input'][1]['content'];text=json.dumps(content)
            self.assertNotIn('0.99',text);self.assertNotIn('Fake success',text)
            self.assertEqual(sum(p['type']=='input_image' for p in content),1)

    def test_reject_changed_images_and_future_or_cross_segment_context(self):
        self.job['composite_sha256']='wrong'
        with self.assertRaises(ValueError):self.body('three')
        self.job['context_event_ids']=['future','0000']
        future={**self.job,'event_id':'future','source_timestamp_ns':101}
        with self.assertRaises(ValueError):
            c.request_body(self.job,'head',{('episode1','0000'):self.job,('episode1','future'):future},self.source,self.three,'gpt-6-astra','medium')

    def test_scores_reject_invalid_values_and_preserve_uncertainty(self):
        for value in [float('nan'),float('inf'),True,-.1,1.1,'0.5']:
            with self.assertRaises(ValueError):c.validate_scores({**self.scores,'progress':value})
        uncertain={**self.scores,'progress':None,'success_probability':None,'status':'uncertain'}
        self.assertEqual(c.validate_scores(uncertain),uncertain)
        with self.assertRaises(ValueError):c.validate_scores({**self.scores,'status':'success'})

    def test_api_protocol_usage_and_no_credentials_in_results(self):
        payload={'id':'response-test','model':'gpt-6-astra','status':'completed','usage':{'input_tokens':100},'output':[
            {'type':'reasoning','summary':[]},{'type':'message','role':'assistant','content':[{'type':'output_text','text':json.dumps(self.scores)}]}]}
        opener=MagicMock();opener.open.return_value.__enter__.return_value=io.BytesIO(json.dumps(payload).encode())
        with patch.dict(os.environ,{'SUB2API_KEY':'fake-key'}),patch.object(c,'build_opener',return_value=opener):
            scores,meta=c.call_api(self.body('head')[0],'https://example.test/v1',10)
        self.assertEqual(scores,self.scores);self.assertEqual(meta['usage']['input_tokens'],100)
        req=opener.open.call_args.args[0]
        self.assertEqual(req.full_url,'https://example.test/v1/responses')
        self.assertEqual(req.get_header('Authorization'),'Bearer fake-key')
        self.assertNotIn('fake-key',json.dumps(meta))
        with patch.dict(os.environ,{},clear=True),patch.object(c,'build_opener') as build:
            with self.assertRaises(ValueError):c.call_api({},'https://example.test/v1',10)
            build.assert_not_called()

    def test_resume_and_configuration_guard(self):
        argv=['compare','--head-dir',str(self.head),'--three-dir',str(self.three),'--output',str(self.out),'--workers','2']
        before={str(p):p.read_bytes() for d in (self.head,self.three) for p in d.glob('*.json*')}
        with patch('sys.argv',argv),patch.dict(os.environ,{'SUB2API_KEY':'fake-key'}),patch.object(c,'call_api',return_value=(self.scores,{'usage':{}})) as call:
            c.main();self.assertEqual(call.call_count,2)
            c.main();self.assertEqual(call.call_count,2)
        self.assertEqual(len(c.read_rows(self.out/'results.jsonl')),2)
        self.assertTrue((self.out/'episode1_report.html').is_file())
        for p,data in before.items():self.assertEqual(Path(p).read_bytes(),data)
        with patch('sys.argv',argv+['--model','different']):
            with self.assertRaises(ValueError):c.main()

    def test_incomplete_and_refused_responses_are_not_scores(self):
        for payload in [
            {'status':'incomplete','output':[]},
            {'status':'completed','output':[{'type':'message','role':'assistant',
              'content':[{'type':'refusal','refusal':'Cannot assess'}]}]},
            {'status':'completed','output':[]},
        ]:
            opener=MagicMock()
            opener.open.return_value.__enter__.return_value=io.BytesIO(json.dumps(payload).encode())
            with patch.dict(os.environ,{'SUB2API_KEY':'fake-key'}),patch.object(c,'build_opener',return_value=opener):
                with self.assertRaises(ValueError):c.call_api({},'https://example.test/v1',10)

    def test_failed_parallel_job_does_not_discard_success_and_resumes(self):
        argv=['compare','--head-dir',str(self.head),'--three-dir',str(self.three),'--output',str(self.out),'--workers','2']
        def reply(body,*args):
            if 'Head camera' in body['input'][1]['content'][0]['text']:raise TimeoutError()
            return self.scores,{'usage':{}}
        with patch('sys.argv',argv),patch.dict(os.environ,{'SUB2API_KEY':'fake-key'}),patch.object(c,'call_api',side_effect=reply):
            with self.assertRaises(SystemExit):c.main()
        self.assertEqual([r['mode'] for r in c.read_rows(self.out/'results.jsonl')],['three'])
        with patch('sys.argv',argv),patch.dict(os.environ,{'SUB2API_KEY':'fake-key'}),patch.object(c,'call_api',return_value=(self.scores,{'usage':{}})) as call:
            c.main();self.assertEqual(call.call_count,1)
        self.assertEqual(len(c.read_rows(self.out/'results.jsonl')),2)


if __name__=='__main__':unittest.main()
