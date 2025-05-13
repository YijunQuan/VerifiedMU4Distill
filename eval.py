from eval_utils import *


def performance_eval(dataset='sst5',teacher_eval=True, purge_eval=True, sisa_eval=True, sst_eval=True, nt=4 ,ns=2, num_epochs=1):
    if teacher_eval:
        teacher_evaluation(dataset=dataset, num_epochs=num_epochs, nt=nt)
    if purge_eval:
        purge_evaluation(dataset=dataset, num_epochs=num_epochs, nt=nt, ns=ns)
    if sisa_eval:
        sisa_evaluation(dataset=dataset, num_epochs=num_epochs,nt=nt, ns=ns)


def time_eval(purge_time=True, sisa_time=True):
    if purge_time:
        purge_time_simulation()
    if sisa_time:
        sisa_time_simulation()

if __name__ == "__main__":    
    if not os.path.isdir('./results/'):
        os.mkdir('./models/')
    performance_eval(dataset='sst5', teacher_eval=True, purge_eval=True, sisa_eval=True, sst_eval=True, num_epochs=1)

    if not os.path.isdir('./results_time/'):
        os.mkdir('./results_time/')
    
    # time_eval(purge_time=False, sisa_time=True)
