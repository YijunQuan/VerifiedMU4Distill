from eval_utils import *


def performance_eval(teacher_eval=True, purge_eval=True, sisa_eval=True):
    if teacher_eval:
        teacher_evaluation()
    if purge_eval:
        purge_evaluation()
    if sisa_eval:
        sisa_evaluation()


def time_eval(purge_time=True, sisa_time=True):
    if purge_time:
        purge_time_simulation()
    if sisa_time:
        sisa_time_simulation()

if __name__ == "__main__":    
    if not os.path.isdir('./results/'):
        os.mkdir('./models/')
    performance_eval()

    if not os.path.isdir('./results_time/'):
        os.mkdir('./results_time/')
    
    time_eval(purge_time=False, sisa_time=True)
