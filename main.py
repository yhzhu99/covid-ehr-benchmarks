import argparse
import pathlib

import torch
from omegaconf import OmegaConf

import signal
import time
import threading

from app import create_app

class MyThread(threading.Thread):
    def __init__(self, target, args=()):
        super(MyThread, self).__init__()
        self.func = target
        self.args = args

    def run(self):
        # 接受返回值
        self.result = self.func(*self.args)

    def get_result(self):
        # 线程不结束,返回值为None
        try:
            return self.result
        except Exception:
            return None

def limit_decor(timeout, granularity):
    """
        timeout 最大允许执行时长, 单位:秒
        granularity 轮询间隔，间隔越短结果越精确同时cpu负载越高
        return 未超时返回被装饰函数返回值,超时则返回 None
    """
    def functions(func):
        def run(*args):
            thre_func = MyThread(target=func, args=args)
            thre_func.setDaemon(True)
            thre_func.start()
            sleep_num = int(timeout//granularity)
            for i in range(0, sleep_num):
                infor = thre_func.get_result()
                if infor:
                    return infor
                else:
                    time.sleep(granularity)
            return None
        return run
    return functions

@limit_decor(600, 10)
def main():
    pathlib.Path("./checkpoints").mkdir(parents=True, exist_ok=True)
    print("===[Start]===")
    parser = argparse.ArgumentParser("Covid-EMR training script", add_help=False)
    parser.add_argument(
        "--cfg", type=str, required=True, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--cuda",
        type=int,
        required=False,
        metavar="CUDA NUMBER",
        help="gpu to train",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="whether to connect database",
    )

    args = parser.parse_args()
    print(f"===[{args.cfg}]===")
    conf = OmegaConf.load(args.cfg)
    conf.db = args.db

    # train on cpu by default
    device = torch.device("cpu")
    if args.cuda is not None:
        device = torch.device(
            f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        )

    create_app(conf, device)
    print("===[End]===")

if __name__ == "__main__":
    main()