(
fork {
    a = {|nL=72, nR=74, dn=0.5, ampL=1, ampR=1, ampC=1|
        var fL = nL.midicps;
        var fR = nR.midicps;
        var fC  = dn.linlin(0, 1, nL, nR).midicps;
        SinOsc.ar(fL) !> Out(0);
        SinOsc.ar(fR) !> Out(2);
        SinOsc.ar(fC) !> Out(1);
    }.play;

    s.sync; // -------------------------

    0.5.wait;

    a.ccbind(81, \nL, _/127*127);
    a.ccbind(82, \nR, _/127*127);
    a.ccbind(83, \dn, _/127);




}
)

