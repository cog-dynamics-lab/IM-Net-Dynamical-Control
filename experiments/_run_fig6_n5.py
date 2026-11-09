import importlib.util, os, time
os.chdir("/mnt/proj/V2027/000")
spec = importlib.util.spec_from_file_location("cco", "experiments/01_cco_simulation.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
t0 = time.time()
k = m.make_fig6(n_init=5)
print(f"[done] kappa_c={k}, took {(time.time()-t0)/60:.1f}min", flush=True)
