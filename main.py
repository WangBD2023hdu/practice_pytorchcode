import argparse
import train

if __name__ == '__main__':
    options = argparse.ArgumentParser()
    options.add_argument("--lr", dest="lr", type=float, default=0.1)
    options.add_argument("--epoch", dest="epoch", type=int, default=100)
    #below parameter is not currently active
    options.add_argument("--path", dest="path", type=str, default="./data" )
    options.add_argument("--cuda", action="store_true", dest="cuda")
    parameter = options.parse_args()
    train.main(parameter.epoch, parameter.lr)

