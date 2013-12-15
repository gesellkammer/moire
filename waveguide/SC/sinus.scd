(
fork {
    a = {|freq=1000, fb=0, wet=0.15, wave=0, envspeed=1, amp=1, delay_ms=1, hiamp=0.2, hifreq=18000, difamp=1|

        //var a0 = SinOsc.ar(freq) * EnvGen.ar(Env.adsr(0, 0.02, sustainLevel:1), Impulse.kr(20));
        var source = Select.ar(wave*3, [Saw.ar(freq), Pulse.ar(freq), PinkNoise.ar]);
        var a0 = ADC(4);
        var a1 = ADC(5);

        var env0 = Amplitude.ar(a0, attackTime:ControlDur.ir*envspeed, releaseTime:ControlDur.ir*envspeed);
        var env1 = Amplitude.ar(a1, attackTime:ControlDur.ir*envspeed, releaseTime:ControlDur.ir*envspeed);

        var hienv0 = (
            BHiPass.ar(a0, hifreq, 0.01)
            !> Amplitude.ar(_, attackTime:ControlDur.ir*envspeed, releaseTime:ControlDur.ir*envspeed)
        );

        var hienv1 = (
            BHiPass.ar(a1, hifreq, 0.01)
            !> Amplitude.ar(_, attackTime:ControlDur.ir*envspeed, releaseTime:ControlDur.ir*envspeed)
        );


        //var source = SinOsc.ar(1000);


        var b0 = (
            a0
            //source * hienv0
            !>< GVerb.ar(_) @wet
        );

        var b1 = (
            a1
            //source * hienv1
            !>< GVerb.ar(_) @wet
        );


        //var b0 = a0 >< (a0*(env0 - env1)) @difamp;
        //var b1 = a1 >< (a1*(env0 - env1)) @difamp;

        b1 = DelayC.ar(b1, delaytime:delay_ms/1000);
        /*
        (
            source
            + (a0*fb)
            + (a1*fb)
            * amp
            // + (Pulse.ar(hifreq)*hiamp)
            !> Out(2)
        );
        */

        b0 !> Out(0);
        b1 !> Out(1);

    }.play;

    s.sync; // --------------------------

    a.ccbind(BCF.fader1, \freq, _.linlin(0, 127, 40, 2000));
    a.ccbind(BCF.fader2, \fb, _.linlin(0, 127, 0, 10));
    a.ccbind(BCF.fader3, \wet);
    a.ccbind(BCF.fader4, \wave);
    a.ccbind(BCF.fader5, \envspeed, _.linlin(0, 127, 0.01, 4));
    a.ccbind(BCF.fader6, \delay_ms, _.linlin(0, 127, 0.1, 20));
    a.ccbind(BCF.fader7, \difamp);

};
)

a.autogui


(
c = {|freq=220|
    Saw.ar(freq) !> Out(0);
}.play
)