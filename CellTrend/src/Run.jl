using CellTrend, Serialization, Random, Plots

# generates figure in manuscript

rstate = Random.Xoshiro(0x69a7dd129e7f60d5, 0x1e7da8f49b6237ba, 0x8317961216fc7d81, 0x2fc455ed74348caf, 0x7db9fb8192742ed2);
d,_=driver_opt(30000; maxiters=10000, save=false, scale=300, rstate=rstate);

# use rstate = nothing to test different random seeds
pl = plot_data(d.T, d; rstate=d.rstate)

# set directory and file name as needed
savefig(pl,"/Users/steve/Desktop/cellTrend.pdf");



##################################################################
##################################################################
##### Various tests, have not checked these for current code version

using CellTrend, Serialization

y,t = CellTrend.rw(300);
v = CellTrend.ema_interp(y,t);
r = 80:0.1:100

CellTrend.ode(rand(3), rand(7), 1, v)

y = CellTrend.rw(300);

######################

d,_ = driver_opt(5000;maxiters=20, save=false, saveat=0.1,
		dir="/Users/steve/Desktop/", learn=0.005);

d,_=driver_opt(30000; maxiters=10000, save=false, scale=300);

######################

# testing
loss_val, yp, y_true, sol, y, p, y_diff, skip = 
		loss(d.p, d.T, d.u0,; saveat=d.saveat, scale=d.scale);
CellTrend.callback(nothing, loss, yp, y_true, sol, y, p, y_diff, skip; direct=false);

##################################################################
##### Old version, must be modified if using on different machine

using CellTrend, Serialization, Random, Plots

rstate = Random.Xoshiro(0x69a7dd129e7f60d5, 0x1e7da8f49b6237ba, 0x8317961216fc7d81, 0x2fc455ed74348caf, 0x7db9fb8192742ed2);
d,outfile=driver_opt(30000; maxiters=10000, save=true, scale=300, rstate=rstate);

# Read results back in and test
dd = deserialize(outfile);
d == dd
# fieldnames(typeof(d))

# Saved run for generating publication analysis and graphics

dir = "/Users/steve/sim/zzOtherLang/julia/projects/Circuits/" * 
			"01_ML_Analogy/CellTrend/output/";
d = deserialize(dir * "SavedRun.jls");

# use rstate = nothing to test different random seeds
pl = plot_data(d.T, d; rstate=d.rstate)
savefig(pl,"/Users/steve/Desktop/cellTrend.pdf");


