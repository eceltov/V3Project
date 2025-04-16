import sys
import datetime
from lib.optimalGridJobScheduler import OptimalGridJobScheduler as Scheduler

# how many annotations should be processed from a single annotation file
annotations_per_config = 20

# prints remaining task count for each job
def debug_print_task_counts():
  job_count = int(sys.argv[2])
  for job_id in range(job_count):
    scheduler = Scheduler(annotations_per_config, job_id, job_count)
    scheduler.debug_print_remaining_task_count()

print(f"Job started at: {datetime.datetime.now()}")

if len(sys.argv) != 3:
  print("Expected two arguments (job ID and number of jobs).")
else:
  job_id = int(sys.argv[1])
  job_count = int(sys.argv[2])

  scheduler = Scheduler(annotations_per_config, job_id, job_count)
  scheduler.do_tasks()
