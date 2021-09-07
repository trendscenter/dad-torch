import torch
import time

<<<<<<< HEAD:dad_torch/power_iteration_BC.py
def power_iteration_BC(B, C, rank=10, numiterations=20, device='cuda', tol=0.0, do_benchmarks=True):
    B = B.float()
    C = C.float()
    if len(B.shape) == 3:
        B = B.reshape(B.shape[0]*B.shape[1], B.shape[2])
    if len(C.shape) == 3:
        C = C.reshape(C.shape[0]*C.shape[1], C.shape[2])
    if len(C.shape) == 1:
        C = C.reshape(1, C.shape[0])
    benchmarks = dict(args=dict(rank=rank, numiterations=numiterations, tol=tol, device=str(device)))
    start = time.time()
    try:
        CC = torch.mm(C.T, C)
    except Exception:
        raise Exception(str(B.shape) + " b - c " + str(C.shape))
    benchmarks["CC"] = str(time.time() - start)
    start = time.time()
=======

def power_iteration_BC(B, C, rank=10, numiterations=20, device='cuda', tol=1e-3):
    CC = torch.mm(C.T, C)
>>>>>>> master:dad_torch/utils/power_iteration_BC.py
    BCC = torch.mm(B, CC)
    benchmarks["BCC"] = str(time.time() - start)
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

    def iterations(computed_eigs=None, is_sigma=1, benchmark_id=0):
        benchmark_key = "iterations_%d" % benchmark_id
        benchmarks[benchmark_key] = dict()
        if computed_eigs is None:
            computed_eigs = []
        if not is_sigma: return zero_result()
        # start with one of the columns
        start = time.time()
        b_k = torch.rand(B.shape[0], device=device)
        benchmarks[benchmark_key]["b_k"] = str(time.time() - start)
        #b_k = B[:, 0]  # np.random.randn(B.shape[0])
        if computed_eigs:
            start = time.time()
            bb, vv = past_values(computed_eigs)
            benchmarks[benchmark_key]["past_values"] = str(time.time() - start)
        start = time.time()
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
        benchmarks[benchmark_key]["iters"] = str(time.time() - start)
        start = time.time()
        sigma = eigenvalue(B, b_k)
        benchmarks[benchmark_key]["sigma"] = str(time.time() - start)
        if torch.isnan(sigma): return zero_result()
        start = time.time()
        c_k = torch.mv(C, torch.mv(B.T, b_k))/sigma
        benchmarks[benchmark_key]["sigma"] = str(time.time() - start)
        if len(computed_eigs)>1 and torch.norm(b_k - computed_eigs[-1]['b'])/torch.norm(computed_eigs[-1]['b'])  < tol:
            r = zero_result()
            computed_eigs[-1]['b'] = r['b']
            computed_eigs[-1]['c'] = r['c']
            computed_eigs[-1]['sigma'] = r['sigma']
            return zero_result()
        return {"b": b_k, "c": c_k, "sigma": sigma, "v": sigma * sigma * b_k}
    eigs = [{"sigma": torch.tensor(1.0, device=device)}]
    for i in range(rank):
        start = time.time()
        eigs += [iterations(computed_eigs=eigs[1:], is_sigma=eigs[-1]["sigma"], benchmark_id=i)]
        benchmarks["rank_%d" % i] = str(time.time() - start)
        if eigs[-1]["sigma"] == 0.0:
            break
    if rank > 2:
        eigs = eigs[1:-2]
    else:
        eigs = eigs[1:]
    #print([x.keys() for x in eigs])
    start = time.time()
    left = torch.stack([x["sigma"] * x["b"] for x in eigs], 1)
    benchmarks["left"] = str(time.time() - start)
    start = time.time()
    right = torch.stack([x["c"] for x in eigs], 1)
    benchmarks["right"] = str(time.time() - start)
    if do_benchmarks:
        return left, right, benchmarks
    return left, right
