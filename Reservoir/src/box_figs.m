(* ::Package:: *)

(* ::Chapter:: *)
(*Reservoir figure for background*)


(* ::Section:: *)
(*Plotting code and test example*)


(* ::Input:: *)
(*CircularNodePlot[inputVector_,valueMatrix_]:=Module[{n=Length[inputVector],m=Length[valueMatrix],plots={},nodeRadius=0.2},(*Create a Viridis-like colorblind-friendly gradient*)colorFunc[x_]:=Blend[{RGBColor[0.267004,0.004874,0.329415],(*Dark purple-blue*)RGBColor[0.253935,0.265254,0.529983],(*Blue*)RGBColor[0.163625,0.471133,0.558148],(*Blue-green*)RGBColor[0.134692,0.658636,0.517649],(*Green*)RGBColor[0.477504,0.821444,0.318195],(*Yellow-green*)RGBColor[0.993248,0.906157,0.143936]   (*Yellow*)},x];*)
(*(*Rest of the code remains the same*)(*Validate input dimensions*)If[Length[valueMatrix[[1]]]!=n,Print["Error: Matrix columns must match input vector length"];*)
(*Return[]];*)
(*(*Function to generate coordinates for nodes on a circle-clockwise from top*)getCirclePoints[numPoints_]:=Table[{Sin[2 Pi k/numPoints],Cos[2 Pi k/numPoints]},{k,0,numPoints-1}];*)
(*(*Function to calculate point on circle edge given center and direction*)edgePoint[center_,direction_]:=center+nodeRadius*Normalize[direction-center];*)
(*(*For each column in the value matrix*)plots=Table[Module[{circlePoints=getCirclePoints[m],arrows,nodes,inputNode,colors},(*Create arrows connecting nodes in a clockwise circle*)arrows=Table[Module[{start=circlePoints[[k]],end=If[k==m,circlePoints[[1]],circlePoints[[k+1]]],startEdge,endEdge},startEdge=edgePoint[start,end];*)
(*endEdge=edgePoint[end,start];*)
(*{Arrowheads[0.065],Arrow[{startEdge,endEdge}]}],{k,1,m}];*)
(*(*Input node position*)With[{inputPos={0,1.7},topNodePos={0,1}},inputNode={{EdgeForm[{Thin,Black}],FaceForm[colorFunc[inputVector[[i]]]],Disk[inputPos,nodeRadius]},{Arrowheads[0.055],(*Added Arrowheads directive*)Arrow[{edgePoint[inputPos,topNodePos],edgePoint[topNodePos,inputPos]}]}}];*)
(*(*Create the circle of nodes with colors from value matrix-clockwise from top*)nodes=Table[{EdgeForm[{Thin,Black}],FaceForm[colorFunc[valueMatrix[[k,i]]]],Disk[circlePoints[[k]],nodeRadius]},{k,1,m}];*)
(*(*Combine all elements with fixed PlotRange to maintain aspect ratio*)Graphics[{arrows,inputNode,nodes},PlotRange->{{-1.6,1.6},{-1.6,2.2}},AspectRatio->Automatic,ImagePadding->None]],{i,1,n}];*)
(*(*Arrange plots in a row with controlled spacing*)GraphicsGrid[{plots},Spacings->{0.2,0}]]*)
(**)
(*(*Example usage:n=4;(*number of columns*)m=6;(*number of nodes in circle*)inputVector={0.2,0.4,0.7,0.9};*)
(*valueMatrix=Table[RandomReal[],{m},{n}];*)
(*CircularNodePlot[inputVector,valueMatrix]*)*)
(**)
(*(* No input matrix and no arrows *)*)
(*CircularNodePlot[valueMatrix_]:=Module[{n=Length[First[valueMatrix]],m=Length[valueMatrix],plots={},nodeRadius=0.2},colorFunc[x_]:=Blend[{RGBColor[0.267004,0.004874,0.329415],RGBColor[0.253935,0.265254,0.529983],RGBColor[0.163625,0.471133,0.558148],RGBColor[0.134692,0.658636,0.517649],RGBColor[0.477504,0.821444,0.318195],RGBColor[0.993248,0.906157,0.143936]},x];*)
(*getCirclePoints[numPoints_]:=Table[{Sin[2 Pi k/numPoints],Cos[2 Pi k/numPoints]},{k,0,numPoints-1}];*)
(*plots=Table[Module[{circlePoints=getCirclePoints[m],nodes},nodes=Table[{EdgeForm[{Thin,Black}],FaceForm[colorFunc[valueMatrix[[k,i]]]],Disk[circlePoints[[k]],nodeRadius]},{k,1,m}];*)
(*Graphics[{nodes},PlotRange->{{-1.6,1.6},{-1.6,2.2}},AspectRatio->Automatic,ImagePadding->None]],{i,1,n}];*)
(*GraphicsGrid[{plots},Spacings->{0.2,0}]]*)


(* ::Input:: *)
(*n=4  (*number of diagrams*)*)
(*m=6  (*nodes per circle*)*)
(*inputVector={0.2,0.4,0.7,0.9}*)
(*valueMatrix=Table[RandomReal[],{m},{n}]*)
(*CircularNodePlot[inputVector,valueMatrix]*)


(* ::Section:: *)
(*Plots for manuscript*)


(* ::Text:: *)
(*Data generated by Julia, Reservoir.jl code*)


(* ::Input:: *)
(*n=5;  (*number of diagrams*)*)
(*m=5;  (*nodes per circle*)*)
(*s=0.1;*)
(*inputVector={0.1, 0.3, 0.5, 0.7, 0.9}^s;*)
(*valueMatrix={{0.099668,0.291313,0.462117,0.604368,0.716298},{0,0.0497928,0.144635,0.227033,0.29331},{0,0,0.0248913,0.0721916,0.113031},{0,0,0,0.012445,0.0360802},{0,0,0,0,0.00622241}}^s;*)
(*p1=CircularNodePlot[inputVector,valueMatrix]*)


(* ::Input:: *)
(*n=5;  (*number of diagrams*)*)
(*m=5;  (*nodes per circle*)*)
(*s=0.3;*)
(*inputVector={0.7, 0.7, 0.7, 0.7, 0.7}^s;*)
(*valueMatrix={{0.604368,0.604368,0.604368,0.604368,0.604368},{0,0.29331,0.29331,0.29331,0.29331},{0,0,0.145613,0.145613,0.145613},{0,0,0,0.0726779,0.0726779},{0,0,0,0,0.036323}}^s;*)
(*p2=CircularNodePlot[inputVector,valueMatrix]*)


(* ::Input:: *)
(*n=5;  (*number of diagrams*)*)
(*m=5;  (*nodes per circle*)*)
(*s=0.1;*)
(*inputVector={0.9, 0.7, 0.5, 0.3, 0.1}^s;*)
(*valueMatrix={{0.716298,0.604368,0.462117,0.291313,0.099668},{0.,0.343583,0.29331,0.227033,0.144635},{0.,0.,0.170121,0.145613,0.113031},{0.,0.,0.,0.0848559,0.0726779},{0.,0.,0.,0.,0.0424025}}^s;*)
(*p3=CircularNodePlot[inputVector,valueMatrix]*)


(* ::Input:: *)
(*n=4;  (*number of diagrams*)*)
(*m=10;  (*nodes per circle*)*)
(*s=0.5;*)
(*valueMatrix={{0.0821635696004539,0.5247732703697224,1.0,0.8485101136402207},{0.29624797343036213,0.4078780709467347,0.6145647542411129,0.36381370297892235},{0.0,0.4279263624993592,0.9450923547976506,0.8587475451516207},{0.6163498938272622,0.6680941576145427,0.6710695053469486,0.41213113794724765},{0.6611761869995688,0.7053023202790354,0.6786089406660549,0.6689960296412525},{0.4881628403396518,0.469229793172753,0.5166155331230053,0.509077526056085},{0.46749702857568115,0.3659029437423945,0.3884079488828662,0.38244195799757535},{0.7331659845726061,0.6889428465442076,0.5799938489706306,0.7496604213539649},{0.22342955391187685,0.2803520114055645,0.5057031442689682,0.7115986821652339},{0.483411731128352,0.6006906706133277,0.7062537267459895,0.551568107534319}}^s;*)
(*p4=CircularNodePlot[valueMatrix]*)


(* ::Input:: *)
(*grid=GraphicsGrid[{{p1},{p2},{p3},{p4}},Spacings->{0,-20},*)
(*Epilog->*)
(*{Black,Arrowheads[0.02],Table[(*Create 4 arrows for each of 3 rows*)Table[With[{(*Scale x positions across available width*)xstart=117+j*111,(*Spread across width 0-600*)(*Scale y positions down the height-600 to 0*)y=-88-i*142     (*Space rows vertically*)},Arrow[{{xstart,y},{xstart+35,y}}]],{j,0,3}  (*4 arrows per row*)],{i,0,2}   (*3 rows*)],*)
(*Text[Style["(a)",14],{32,-21}],*)
(*Text[Style["(b)",14],{32,-163}],*)
(*Text[Style["(c)",14],{32,-305}],*)
(*Text[Style["(d)",14],{32,-462}],*)
(*Text[Style["rising",14],{77.5,-526}],*)
(*Text[Style["flat",14],{226,-526}],*)
(*Text[Style["falling",14],{374.5,-526}],*)
(*Text[Style["sine",14],{523,-526}]*)
(*}*)
(*]*)
(**)


(* ::Input:: *)
(*Export["/Users/steve/Desktop/reservoir.pdf", grid];*)
