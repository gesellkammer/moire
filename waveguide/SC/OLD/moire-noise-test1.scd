// acoustical moire test
(
fork {
    a = {|freqshift=2, phase_shift=0, amp0=0.01, amp1=0.01, delay_ms=0, freq=100|
        //var source = SinOsc.ar(freq);
        var source0 = PinkNoise.ar;
        //var source1 = WhiteNoise.ar;
        var source1 = PinkNoise.ar;
        var a0 = source0 * amp0.lag(0.5);
        var a1 = source1 * amp1.lag(0.5) !> FreqShift.ar(_, freqshift, phase_shift) !> DelayC.ar(_, 0.2, delay_ms/1000);
        Out.ar(0, [a0+a1]);
        Out.ar(2, [a0*a1]);
        0;
    }.play;

    s.sync; // ---------------------------------------

    a.ccbind(BCF.fader_cc(1), \amp0);
    a.ccbind(BCF.fader_cc(2), \amp1);
    a.ccbind(BCF.fader_cc(3), \freqshift, _/127*12, post:31415);
    a.ccbind(BCF.fader_cc(4), \phase_shift, _/127*2pi, post:31415);
    a.ccbind(BCF.fader_cc(5), \delay_ms, _/127*30, post:31415);
    a.ccbind(BCF.fader_cc(6), \freq, _/127*400, post:31415);

}

)