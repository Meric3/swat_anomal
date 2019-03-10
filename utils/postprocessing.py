import torch
from torch.autograd import Variable
import numpy as np

def get_anomalscore_encdec(base_model, generate_batchfy, length,args):
    outSeq = []
    scores = []
    base_model.encoder.eval()
    base_model.decoder.eval()
    hidden_enc = base_model.encoder.init_hidden(1)
    endPoint = length #446000 
    step = args['seq_length']
    feature_dim = generate_batchfy.size(-1)

    for i in range(0,endPoint,step):
        Outputseq_enc, hidden_enc = base_model.encoder.forward(Variable(generate_batchfy[i:i+step]).cuda(), hidden_enc)
        deccoder_input = Variable(torch.zeros(Outputseq_enc.size())).cuda()
        deccoder_input[0,:,:] = Outputseq_enc[-1,:,:]
        try:
            deccoder_input[1:,:,:] = generate_batchfy[i+step:i+step+step-1,:,:]
        except:
            continue
        Outputseq_enc, hidden_enc = base_model.decoder.forward(deccoder_input, hidden_enc, return_hiddens=True)
        if Outputseq_enc.size()[0] != generate_batchfy[i:i+step].size()[0] :
            continue
        error = torch.abs(torch.add( Outputseq_enc.view(-1,generate_batchfy.size(-1)).cpu(), \
                                -generate_batchfy[i:i+step].view(-1,generate_batchfy.size(-1)).cpu()))       
        outSeq.append(error)
        
    outSeq = torch.cat(outSeq,dim=0)
    mean = outSeq.mean(dim=0)
    conv_inv = torch.inverse(torch.from_numpy(np.cov((outSeq-mean).t().detach().numpy()))).type(torch.FloatTensor)
    print("anomal_detection...")
    
    for i in range(0,endPoint):
        score = torch.mm(torch.add(outSeq[i], -mean.view(feature_dim,-1).t()),  torch.mm(conv_inv,  torch.add(outSeq[i], -mean.view(feature_dim,-1))))
        scores.append(score.view(-1,generate_batchfy.size(-1)))
    scores = torch.cat(scores,dim=0)
    scores = scores.detach().numpy()
    nomalize_scores =  (scores - scores.mean(axis=0))/scores.var(axis=0)
    return nomalize_scores



def get_anomalscore_minmax(base_model, generate_batchfy, length,args):
    outSeq = []
    scores = []
    base_model.encoder.eval()
    base_model.decoder.eval()
    hidden_enc = base_model.encoder.init_hidden(1)
    endPoint = length #446000 
    step = args['seq_length']
    feature_dim = generate_batchfy.size(-1)

    for i in range(0,endPoint,step):
        Outputseq_enc, hidden_enc = base_model.encoder.forward(Variable(generate_batchfy[i:i+step]).cuda(), hidden_enc)
        deccoder_input = Variable(torch.zeros(Outputseq_enc.size())).cuda()
        deccoder_input[0,:,:] = Outputseq_enc[-1,:,:]
        try:
            deccoder_input[1:,:,:] = generate_batchfy[i+step:i+step+step-1,:,:]
        except:
            continue
        Outputseq_enc, hidden_enc = base_model.decoder.forward(deccoder_input, hidden_enc, return_hiddens=True)
        if Outputseq_enc.size()[0] != generate_batchfy[i:i+step].size()[0] :
            continue
        error = torch.abs(torch.add( Outputseq_enc.view(-1,generate_batchfy.size(-1)).cpu(), \
                                -generate_batchfy[i:i+step].view(-1,generate_batchfy.size(-1)).cpu()))       
        outSeq.append(error)
        
    outSeq = torch.cat(outSeq,dim=0)
    outSeq = outSeq.detach().numpy()
    return outSeq - outSeq.min(axis=0) / (outSeq.max(axis=0) - outSeq.min(axis=0) )
#     return outSeq - outSeq.min(dim=0)[0] / (outSeq.max(dim=0)[0]  - outSeq.min(dim=0)[0] )




def evaluate_conv(nomalize_scores, num_samples, conv, check_step,attack_list ,length,prints,args):
    check_step = check_step
    endPoint = length
    
    norm = LA.norm(nomalize_scores, axis=1)
    
    if conv != 0 :
        gauss_kernel = Gaussian1DKernel(conv)
        norm = convolve(norm, gauss_kernel)

    
    maximum = norm.max()
    th = np.linspace(0, maximum, num_samples)
    precision_ = []
    recall_ = []
    score_ = []
    attack_list_ = []
    for j in range(check_step):
        anomaly = (norm > th[j])
        anomaly = anomaly.astype(np.int16)
        pr = precision_score(args['test_y'][:endPoint].cpu().numpy(),anomaly)
        re = recall_score(args['test_y'][:endPoint].cpu().numpy(),anomaly)
        precision_.append(pr)
        recall_.append(re)
        score_.append(2/((1/re) + (1/pr)))
        attack_list_acc = []
        for idxx in range(0, len(attack_list)):
            k, t = np.unique(anomaly[attack_list['Start Time'][idxx]:attack_list['End Time'][idxx]], return_counts=True)
            if t.shape[0] > 1:
                attack_list_acc.append(t[1]/(t[0] + t[1]))
            elif k == 1:
                attack_list_acc.append(1.0)
            elif k == 0:
                attack_list_acc.append(0.0)
            else:
                print("faral error in evaluation")
        attack_list_.append(attack_list_acc)

    tp = 0
    idx = 0
    for i in range(len(score_)):
        if tp < score_[i]:
            tp = score_[i]
            idx = i
    last_anomal = anomaly = (norm > th[idx])
    last_anomal = last_anomal.astype(np.int16)
    
    zerolist = 0
    for nonzero in range(len(attack_list_[idx])):
        if attack_list_[idx][nonzero] == 0.0:
            zerolist = zerolist + 1            
            
    if prints == True :
        print('-'*100)
        print("conv : ",conv,"idx : ",idx,"recall : ", recall_[idx], "precision : ",precision_[idx], "F1 measure : ",score_[idx], " accure_attack_num : ", 36 - zerolist )
    return recall_, precision_, score_, attack_list_, idx, last_anomal