import pydoc
from torchsummary import summary
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('model', help='Path to model')
    parser.add_argument('input_size', help='Input image size', type=int)
    parser.add_argument('batch_size', help='Size of mini-batch', type=int)
    args = parser.parse_args()

    model = pydoc.locate(args.model)()
    input_size = args.input_size
    batch_size = args.batch_size

    summary(model, (1, input_size, input_size), batch_size=batch_size)

if __name__ == '__main__':
    main()