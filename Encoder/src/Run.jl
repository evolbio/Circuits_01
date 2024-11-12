using Encoder, Random, Lux, Plots, Serialization, Dates

# start Julia from Encoder directory, with output as direct subdirectory

# either use saved results as in first block, or generate those results
# by running code in second block, should yield identical "d" variable

#######################################################################
# used saved results
d = deserialize("output/node1_25K_v11.1.jls");	# created by Julia V 11.1
T = d.T;
#######################################################################
# or generate results, takes a few hours

rseed = 19;				# 0 => random seeding
input = rw;				# function to create input time series
T = 300;				# saveat = 0.1, so n=10*T
max_iter = 25000;
offset = os = 1;		# if using > 1, then check code, may not work
nodes = 1;
alpha = 0.2;
adam = 0.005;

l,y,yp,m,p,s,fs,g=Encoder.driver(T; rseed=rseed, adam=adam, log2_nodes=nodes,
					rtol=1e-9, max_iter=max_iter, offset=os, alpha=alpha,
					ps=nothing, st=nothing);
d = (rseed=rseed, input=input, T=T, os=os, max_iter=max_iter, nodes=nodes,
		alpha=alpha, adam=adam, l=l, y=y, yp=yp, m=m, p=p, s=s, g=g, fs=fs,
		note="", git="0b7945c");
		
# optionally, save newly generated results
date = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS");
serialize(dir * "output/$date.jls", d);

#######################################################################
# make figures, should match publication figures

# Figure 3
r1 = Xoshiro(0x438ca6f9093ac7e8, 0xec33bb9ce245df88, 0x472f8fc901466a9a, 0xbdeb83c599cc95d7, 0xf351496ae440146a);
pl = plot_data(T,d; rstate = r1)
savefig(pl, "/Users/steve/Desktop/enc_dyn.pdf");

# Figure 5
r1 = Xoshiro(0x93612334bbdaa533, 0x67f1474fa2334987, 0x7f005ac01fe0fd8b, 0x59aa5a1eaf7f4e41, 0xcf6ca428078f5bbe);
pl = plot_err_distn(T,d; n=10000, rstate = r1, match_test=100000)
savefig(pl, "/Users/steve/Desktop/enc_err_distn.pdf");

# Figure 6
logfile = "/Users/steve/sim/zzOtherLang/julia/projects/Circuits/" * 
		"01_ML_Analogy/Encoder/output/model.log";
r1 = Xoshiro(0xc00aa92ef0d250d8, 0xfd26795c61771856, 0x8c3178f5b28be4eb, 0x2523fd958fda82a7, 0xd4e4e894f438ea36);
st,layer,y_pred=log_dynamics(T,d; rstate=r1, logfile=logfile);
# plot_log_all(d,st,layer,y_pred)
pl = plot_log_node1(d,st,layer,y_pred)
savefig(pl, "/Users/steve/Desktop/enc_states.pdf");

