import torch.nn as nn
import numpy as np
import torch.cuda.amp as amp


#################### Define get_neuron ########################
def get_neuron(template, latitudes, longitudes):
    idx_lats=[]
    for lat in latitudes:
        idx_lats.append(np.where(template.latitude==lat)[0][0])
    idx_lons=[]
    for lon in longitudes:
        idx_lons.append(np.where(template.longitude==lon)[0][0])
    return idx_lats, idx_lons
#################### Define combinedModel #####################
class combinedModel(nn.Module):
  def __init__(self, model, forecast_step, idx_vars, idx_lats, idx_lons, mean, sigma):
    super(combinedModel, self).__init__()
    self.model=model
    self.forecast_step=forecast_step
    self.idx_vars=idx_vars
    self.idx_lats=idx_lats
    self.idx_lons=idx_lons
    self.mean=mean
    self.sigma=sigma
  def forward(self, x):
    with amp.autocast(True):
        x=(x-self.mean)/self.sigma
        for i in range(int(self.forecast_step)):
            x=self.model(x)
        x=x*self.sigma+self.mean
        x0=x[:,self.idx_vars,self.idx_lats[0]:self.idx_lats[-1],self.idx_lons[-1], None]
        x_west=x[:,self.idx_vars,self.idx_lats[0]:self.idx_lats[-1],self.idx_lons[0]:self.idx_lons[-2]]
        x_cynthia=torch.cat((x0,x_west), axis=3)
        ec_wind=(x_cynthia[:,0,:,:]**2+x_cynthia[:,1,:,:]**2)
        x_cynthia=0.5*torch.mean(ec_wind)
    return x_cynthia

