import numpy as np
import torch


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
class combinedModel(torch.nn.Module):
  def __init__(self, model, forecast_step, idx_lats, idx_lons, mean, sigma, x_target, cf_type='ivt'):
    super(combinedModel, self).__init__()
    self.model=model
    self.forecast_step=forecast_step
    self.idx_lats=idx_lats
    self.idx_lons=idx_lons
    self.mean=mean
    self.sigma=sigma
    self.cf_type=cf_type
    self.x_target=x_target  # target analysis

  def forward(self, x):
    with torch.cuda.amp.autocast(True):
        x=(x-self.mean)/self.sigma
        for i in range(int(self.forecast_step)):
            x=self.model(x)
        x=x*self.sigma+self.mean
        x_region=x[:,:,self.idx_lats[0]:self.idx_lats[-1],self.idx_lons[0]:self.idx_lons[-1]]
        x_region_target=self.x_target[:,:,self.idx_lats[0]:self.idx_lats[-1],self.idx_lons[0]:self.idx_lons[-1]]
        if self.cf_type == 'surface_ke':
            cf = self.surface_ke(x_region, x_region_target)
        elif self.cf_type == 'ivt':
            cf = self.ivt(x_region, x_region_target)
        else:
            print('Unknown cost function: '+self.cf_type+'\n')
    return cf

  def surface_ke(self, x_predicted, x_target):
    ke_target =  (x_target[:,0,:,:]**2+x_target[:,1,:,:]**2)
    ke_predicted = (x_predicted[:,0,:,:]**2+x_predicted[:,1,:,:]**2)
    cf=0.5*torch.mean(ke_target - ke_predicted)
    return cf

  def ivt(self,x_predicted, x_target):
    g=9.81 #m/s/s

    ua = x_predicted[:,8:8+13,:,:]
    va = x_predicted[:,8+13:8+13*2,:,:]
    qq = x_predicted[:,(8+13*4):(8+13*5),:,:]

    Fu = torch.sum(ua*qq/g,dim=1)*100     #this '*100' is because the vertical coordinate is in hPa rather than Pa
    Fv = torch.sum(va*qq/g,dim=1)*100

    ivt_predicted = ((Fu**2 + Fv**2)**(1/2))

    ua = x_target[:,8:8+13,:,:]
    va = x_target[:,8+13:8+13*2,:,:]
    qq = x_target[:,(8+13*4):(8+13*5),:,:]

    Fu = torch.sum(ua*qq/g,dim=1)*100     #this '*100' is because the vertical coordinate is in hPa rather than Pa
    Fv = torch.sum(va*qq/g,dim=1)*100

    ivt_target = ((Fu**2 + Fv**2)**(1/2))

    cf=torch.mean(ivt_target - ivt_predicted)
    return cf


