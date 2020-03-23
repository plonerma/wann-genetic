from .task import IrisTask
from .performance import Performance

available_tasks = {
    'iris': IrisTask
}

def select_task(task_name):
    return available_tasks.get(task_name, IrisTask)
