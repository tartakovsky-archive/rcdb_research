import sys
import json

from commons.features.parallel_calc_all import JobHandler


if __name__ == '__main__':
    job_meta = json.loads(sys.argv[1])
    job_handler = JobHandler(job_meta)
    job_handler.run_job()
