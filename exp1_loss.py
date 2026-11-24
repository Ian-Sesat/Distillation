import torch 
import torch.nn.functional as F

def kl_term(P_teacher, Q_student):
    logP = torch.log(P_teacher + 1e-9)
    logQ = torch.log(Q_student + 1e-9)

    kl_elements = P_teacher * (logP - logQ)      
    kl_per_row = kl_elements.sum(dim=1)          

    return kl_per_row.sum()      

def S_tau(U, V, tau):
    logits = (U @ V.t()) / tau
    return F.softmax(logits, dim=1)

def clip_anchor(s_img, s_txt, logit_scale=100.0):
    logits = logit_scale * s_img @ s_txt.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))           

def exp1_loss(s_img, s_txt, t_img, t_txt, tau_s=0.07, tau_t=0.07, lam=1.0, logit_scale=100.0):
    s_it = S_tau(s_img, s_txt, tau_s) 
    t_it = S_tau(t_img, t_txt, tau_t)   
    s_ti = S_tau(s_txt, s_img, tau_s);  
    t_ti = S_tau(t_txt, t_img, tau_t)  

    L_it = kl_term(t_it, s_it)
    L_ti = kl_term(t_ti, s_ti)
    b = s_img.size(0)
    L_distill = (L_it + L_ti) / (2 * b)       

    if lam < 1.0:
        L_clip = clip_anchor(s_img, s_txt, logit_scale)
    else:
        L_clip = torch.tensor(0.0, device=s_img.device)

    total = (1 - lam) * L_clip + lam * L_distill
    return total, {"L_it": L_it.item(), "L_ti": L_ti.item(),
                   "L_distill": L_distill.item(), "L_clip": L_clip.item()}

