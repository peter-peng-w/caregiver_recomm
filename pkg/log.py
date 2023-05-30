from datetime import datetime


def log(*args, timer=None):
    time = (datetime.now() if timer is None else timer.now()).strftime('%Y-%m-%d %H:%M:%S')
    print('[RECOMM]', f'{time}    ', *args)

    try:
        with open('recomm_logs.txt', 'a+') as log_file:
            print('[RECOMM]', f'{time}    ', *args, file=log_file)
    except Exception:
        print('Unable to save logs in log.txt')
