import asyncio
import unittest
from ai_workload_platform.api import WorkloadService
from ai_workload_platform.models import ResourceRequest, Workload, WorkloadState
from ai_workload_platform.scheduler import LocalScheduler

class WorkloadTests(unittest.TestCase):
    def test_submit_and_quota(self):
        service = WorkloadService(LocalScheduler(), ResourceRequest(4, 8192, 1))
        job = Workload("job-1", "default", ResourceRequest(2, 1024, 1))
        self.assertEqual(asyncio.run(service.submit(job)).state, WorkloadState.RUNNING)
        with self.assertRaises(ValueError):
            asyncio.run(service.submit(Workload("job-2", "default", ResourceRequest(2, 1024, 2))))
