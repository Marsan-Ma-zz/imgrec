import argparse

def params_setup(cmdline=None):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--model_name', type=str, required=True, help='training sample name')
  parser.add_argument('--label_size', type=int, required=True, help='label size of training sample')
  parser.add_argument('--gpu_usage', type=float, default=0.5, help='tensorflow gpu memory fraction used')
  parser.add_argument('--img_size', type=int, default=227, help='image size, also model input size')



  if cmdline:
    args = parser.parse_args(cmdline)
  else:
    args = parser.parse_args()


  args.down_sampling = {str(n): 10000 for n in range(13)}

  return args

