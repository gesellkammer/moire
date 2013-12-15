(
a = {|param100=10, freq0=120, fb=80|
    var param = param100/100 !> Lag.kr(_, 0.1);
    var a0 = Saw.ar(freq0.lag(0.1));
    var a0_amp = 1;
    var pre_tube  = (a0*a0_amp) + (LocalIn.ar(1)*(fb/100));
    var delay_ms  = param.linlin(0, 1, 0, 10);
    var post_tube = DelayC.ar(pre_tube, delaytime:delay_ms/1000);
    var wet = param;

    post_tube !> LocalOut.ar(_);
    a0 >< post_tube @wet !> Out.ar(0, _);

}.play
)



a.autogui