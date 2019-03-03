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
