import torch


def power_iteration_BC(B, C, rank=10, numiterations=20, device='cuda', tol=1e-3):
    CC = torch.mm(C.T, C)
    BCC = torch.mm(B, CC)

    def zero_result():
        sigma = torch.tensor(0.0, device=device)
        b_k = torch.zeros(B.shape[0], device=device)
        c_k = torch.zeros(C.shape[0], device=device)
        return {"b": b_k, "c": c_k, "sigma": sigma, "v": b_k}
        
    def eigenvalue(B, v):
        Bv = torch.mv(B.T, v)
        return torch.sqrt(Bv.dot(torch.mv(CC, Bv)))

    def past_values(computed_eigs):        
        bb = torch.stack([x['b'] for x in computed_eigs], 0)
        vv = torch.stack([x['v'] for x in computed_eigs], 0)
        return bb, vv
    
    def iterations(computed_eigs=[], is_sigma=1):
        if not is_sigma: return zero_result()
        # start with one of the columns
        b_k = torch.rand(B.shape[0], device=device)
        #b_k = B[:, 0]  # np.random.randn(B.shape[0])
        if computed_eigs:
            bb, vv = past_values(computed_eigs)
        for _ in range(numiterations):
            adjuster = torch.tensor(0.0, device=device)
            if computed_eigs:
                adjuster = torch.mv(vv.T, torch.mv(bb, b_k))
            # calculate the matrix-by-vector product (BC'CB' - adjusting_matrix)b
            b_k1 = torch.mv(BCC, torch.mv(B.T, b_k)) - adjuster
            # calculate the norm of b
            b_k1_norm = torch.norm(b_k1)            
            # re normalize the vector
            b_k = b_k1 / b_k1_norm
            
        sigma = eigenvalue(B, b_k)
        if torch.isnan(sigma): return zero_result()
        c_k = torch.mv(C, torch.mv(B.T, b_k))/sigma
        if len(computed_eigs)>1 and torch.norm(b_k - computed_eigs[-1]['b'])/torch.norm(computed_eigs[-1]['b'])  < tol:
            r = zero_result()
            computed_eigs[-1]['b'] = r['b']
            computed_eigs[-1]['c'] = r['c']
            computed_eigs[-1]['sigma'] = r['sigma']
            return zero_result()
        return {"b": b_k, "c": c_k, "sigma": sigma, "v": sigma * sigma * b_k}
    eigs = [{"sigma": torch.tensor(1.0, device=device)}]
    for i in range(rank):
        eigs += [iterations(computed_eigs=eigs[1:], is_sigma=eigs[-1]["sigma"])]
        if eigs[-1]["sigma"] == 0.0:
            break
    eigs = eigs[1:-2]
    return (
        torch.stack([x["sigma"] * x["b"] for x in eigs], 1),
        torch.stack([x["c"] for x in eigs], 1),)
