################ LICENSE ######################################
# This software is Copyright © 2024 The Regents of the University of California.
# All Rights Reserved. Permission to copy, modify, and distribute this software and its documentation
# for educational, research and non-profit purposes, without fee, and without a written agreement is
# hereby granted, provided that the above copyright notice, this paragraph and the following three paragraphs
# appear in all copies. Permission to make commercial use of this software may be obtained by contacting:
#
# Office of Innovation and Commercialization 9500 Gilman Drive, Mail Code 0910 University of California La Jolla, CA 92093-0910 innovation@ucsd.edu
# This software program and documentation are copyrighted by The Regents of the University of California. The software program and documentation are
# supplied “as is”, without any accompanying services from The Regents. The Regents does not warrant that the operation of the program will
# be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason.
#
# IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER
# IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
# UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
################################################################
#################### Import libraries ##########################
## Import libraries
import os
import sys
import numpy as np
import torch
from sfno.load_sfno import load_sfno
from utils.cost_function import combinedModel, get_neuron
from utils.sfno_utils import array_to_xarray, load_ic

exec(open('sfno/sfnonet.py').read())


#################### Define working directory ##################
## Set the working directory
if len(sys.argv) > 1:
    workdir=sys.argv[1]
else:
    workdir='./'

#################### Define setup parameters ##################
modelName='sfno'
sfno_vars=["uas","vas","u100","v100","tas","sp","mslp","tcwv",
           "ua50","ua100","ua150","ua200","ua250","ua300","ua400","ua500","ua600","ua700","ua850","ua925","ua1000",
           "va50","va100","va150","va200","va250","va300","va400","va500","va600","va700","va850","va925","va1000",
           "z50","z100","z150","z200","z250","z300","z400","z500","z600","z700","z850","z925","z1000",
           "ta50","ta100","ta150","ta200","ta250","ta300","ta400","ta500","ta600","ta700","ta850","ta925","ta1000",
           "hur50","hur100","hur150","hur200","hur250","hur300","hur400","hur500","hur600","hur700","hur850","hur925","hur1000"]
################################################################################
date_peak=np.datetime64('2022-12-27T12:00:00')
longitudes=list(np.arange(235.,246+0.25,0.25))
latitudes=np.arange(32,42+0.25,0.25)

lead_times=[24, 36, 48]

################################################################################
# Device
#torch.backends.cudnn.benchmark=True
#device=torch.device('cuda') if torch.cuda.is_available() else 'cpu'
device='cpu'

##########################################
# Neural weather model
# Download the .tar file from the ECMWF ai-models Github repository: https://github.com/ecmwf-lab/ai-models
# Create a folder called "models" in your working directory and store the .tar file here
# The load_sfno function is available at as "load_model":
model_sfno=load_sfno(workdir+'/sfno_data/weights.tar', device)

##########################################
for lead_time in lead_times:
    date_ic=date_peak-np.timedelta64(lead_time, 'h')
    print('Lead time: '+str(lead_time)+' at date'+str(date_ic))

    ##########################################
    # Initial condition
    x, mean_arr, sigma_arr, data_ic = load_ic(date_ic=date_ic, sfno_vars=sfno_vars, workdir=workdir, device=device)
    # Get output neuron of interest
    idx_lats, idx_lons=get_neuron(template=data_ic,latitudes=latitudes,longitudes=longitudes)

    # Define new model comprising >1 forecast step
    # Surface winds have indices [0,1] in the variable list
    model=combinedModel(model_sfno, forecast_step=lead_time/6, idx_vars=[0,1], idx_lats=np.flip(idx_lats), idx_lons=idx_lons, mean=mean_arr, sigma=sigma_arr)
    # Set "evaluation" mode and send to device
    model=model.eval()
    model=model.to(device)
    ##########################################
    # Compute the sensitivities (matrix)

    g=torch.autograd.grad(outputs=model(x), inputs=x, retain_graph=True)[0]
    # Build xarray object
    g=array_to_xarray(g, sfno_vars, data_ic[sfno_vars[0]].dims, data_ic)
    # Save the sensitivities
    g.to_netcdf(workdir+'/data/'+modelName+'/gradients/gradients-lt'+str(lead_time)+'.nc')
#############################################
