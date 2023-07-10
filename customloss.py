import numpy as np
import torch
from hyperparameters import MAXDBS
import torch.nn as nn
import random
from sklearn.metrics import classification_report
import pandas as pd

# add muliplication by nj charge to the potential matrix
def get_potential(x, b, gridsize):
    epbs_matrix = torch.load("/home/igardner/SiDBTransformer/electric_potential_matrix.pth") #size 42, 42, 1764
    x = x.squeeze(-1)

    epbs_out = torch.zeros(b, gridsize, gridsize, MAXDBS)
    for batch_index, batch in enumerate(x):
        batchnz = torch.nonzero(batch)
        for i, currenti in enumerate(batchnz):
            for j, comp_cord in enumerate(batchnz[i:]):
                loc = comp_cord[0] * 42 + comp_cord[1]
                epbs_out[batch_index][currenti[0]][currenti[1]][j] = epbs_matrix[currenti[0]][currenti[1]][loc]
                loc2 = currenti[0] * 42 + currenti[1]
                epbs_out[batch_index][comp_cord[0]][comp_cord[1]][i] = epbs_matrix[comp_cord[0]][comp_cord[1]][loc2]

    epbs_out = torch.sum(epbs_out, dim=-1)
    epbs_out = torch.nn.functional.normalize(epbs_out, p=2.0, dim=-1)
    return epbs_out


#ELECTRIC_CHARGE = - 1.602 * 1E-19

def get_energy(inputs, mask):
    pred = inputs.argmax(dim=-1, keepdim=True)
    pred[~mask] = 0
    charge_pred_ni = torch.where(pred == 1, -1, pred)
    local_potential = get_potential(charge_pred_ni, for_energy=True)
    charge_pred = charge_pred_ni.repeat(1, 1, 1, local_potential.shape[-1]) #64, 42, 42, MAXDBS - -1 charge repeated 0s repeated
    local_potential = local_potential * 0.5 #* ELECTRIC_CHARGE #positive electric charge to counteract negative charge at beginning?
    energy_m = torch.mul(charge_pred, local_potential) #local potential = Vij * nj charge, here local_potential * ni charge
    energy_m = torch.sum(energy_m, dim=(1, 2, 3))
    #energy_m *= -1
    return energy_m # in electron volts not jules hence no electric charge


def sig(x):
    return 1 / (1 + np.exp(-x))


#def get_loss(outputs, targets):
 #   lossfn = nn.CrossEntropyLoss()
    # energymask = targets >= 0
   # energy = get_energy(outputs, energymask)
  #  targets = targets.reshape(-1)
   # mask = targets >= 0
    #outputs = outputs.view((-1, 2))
    #masked_target = targets[mask]
    #masked_output = outputs[mask.unsqueeze(-1).repeat(1, 2)].view(-1, 2)
    #loss = lossfn(masked_output, masked_target)
    #loss += loss * sig(torch.sum(energy))
    #return loss


def get_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report

