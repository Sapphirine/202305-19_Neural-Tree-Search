import jax 
import haiku as hk 

class Decoder(hk.Module):
  def __init__(self, name='decoder'):
    super().__init__(name=name)
    
    self.deconv = hk.Sequential(
        [hk.Flatten()]
      + [hk.Linear(1024), jax.nn.relu]
      + [hk.Linear(4096), jax.nn.relu]
      + [hk.Reshape(output_shape=(8, 8, 64))]
      + [hk.Conv2DTranspose(32, kernel_shape=4, stride=2, padding='SAME'), jax.nn.relu]
      + [hk.Conv2DTranspose(16, kernel_shape=4, stride=2, padding='SAME'), jax.nn.relu]
      + [hk.Conv2DTranspose(2, kernel_shape=4, stride=2, padding='SAME')]
    )

  def __call__(self, s):
    ds = self.deconv(s)
    # proj_pred = ds[:, :, :, :1]
    # fp_exp_pred = ds[:, :, :, 1:]
    return ds


class PoseEstimator(hk.Module):
  def __init__(self, name='pose'):
    super().__init__(name)

    self.pose_conv = hk.Sequential(
      [hk.Conv2D(64, kernel_shape=4, stride=2, padding='SAME'), jax.nn.relu]
    + [hk.Conv2D(32, kernel_shape=4, stride=2, padding='SAME'), jax.nn.relu]
    + [hk.Conv2D(16, kernel_shape=3, stride=1, padding='SAME'), jax.nn.relu]
    + [hk.Flatten()]
    + [hk.Linear(1024), jax.nn.relu]
    )

    self.x_head = hk.Sequential([
      hk.Linear(128), 
      jax.nn.relu,
      hk.Linear(1)
      ])
    
    self.y_head = hk.Sequential([
      hk.Linear(128), 
      jax.nn.relu,
      hk.Linear(1)
      ])

    self.o_head = hk.Sequential([
      hk.Linear(128), 
      jax.nn.relu,
      hk.Linear(1)
      ])
  
  def __call__(self, ds):
    p = self.pose_conv(ds)
    dx = self.x_head(p)
    dy = self.y_head(p)
    do = self.o_head(p)
    pose_pred = jax.lax.concatenate([dx,dy,do], dimension=1)
    return pose_pred
  

def _init_resnet_decoder_func(decoder_module):
  def decoder_func(s):
    decoder_model = decoder_module()
    return decoder_model(s)
  return decoder_func

def _init_resnet_pose_func(pose_module):
  def pose_func(pose_input):
    pose_model = pose_module()
    return pose_model(pose_input)
  return pose_func