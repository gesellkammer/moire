/*
README
======

This is the main patch for __________, by Eduardo Moguillansky

1) Go down to SETUP and modify as indicated to fit your setup
2) There are two code-blocks to evaluate to get the patch started
   You evaluate a code-block by placing your cursor at the _____
   in each "EVAL THIS" sign, and pressing Cmd-ENTER.
   Eval each of the code-blocks
3) There is no step 3
*/

/*

===================================================
=                                                 =
=                     SETUP                       =
=                                                 =
===================================================

*/

(
// -------------------------------------------------------------------
//                     EVAL THIS: ________________
// --------------------------------------------------------------------

// Base path of the project. Change it when moving it to another place.
// This should reflect the path where this file was saved
~basepath = "~/proj/grame".absolutePath;   // <------------------- CHANGE THIS WHEN MOVING TO ANOTHER PLACE!!

~readsettings = {|path|
    path = path.absolutePath;
    if(File.exists(path)) {
        var file = File(path, "r");
        var str = file.readAllString;
        var dict = str.interpret;
        assert { dict.isKindOf(Dictionary) };
        dict;
    }{
        ();  // <---- defaults to an empty Event
    };
};

~savesettings = {|dict, path|
    var file = File(path.absolutePath, "w");
    var str = dict.asCompileString;
    file.write(str);
    file.close;
    // ("settings saved at" + path).postln;
};

~setApath = ~basepath +/+ "CONFIG" +/+ "setA.txt";

~conf = ();
~conf.micb3_fftsize = 4096;
~conf.micb3_bufpath = ~basepath +/+ "ASSETS" +/+ "A4174cm.wav";
~conf.micb3_buf.value.free;
// ~conf.micb3_buf = ~resonator_analisys.(~conf.micb3_bufpath, ~conf.micb3_fftsize);
~conf.micb3_buf = PartConv.getBuffer(~conf.micb3_bufpath, ~conf.micb3_fftsize);

Elib.midi_init;
)

(  // ____________________________________________________________________________________________________________________________
fork {
    var synthargs, fbackset, fbackcurve, pedalcurve, ccbind, distortion_curve, setvalue;
    var printaddr = NetAddr("localhost", 31415);

    Elib.midi_init;

    if( ~conf.isNil ) {
        "\n>>>>>>>>>>> evaluate SETUP first! <<<<<<<<<<<<<\n".throw;
    };

    if( ~conf.micb3_buf.value.isNil ) {
        "+--------------------------+".postln;
        "| WARNING: buffer not set! |".postln;
        "+--------------------------+".postln;

    };

    ~setA = ~readsettings.(~setApath).postln;
    ~ccbinds = List();

    ~ccbinds_reflect = {
        var mout = MIDIOut.newMatching("BCF");
        ~ccbinds.do {|bind|
            var cc, label, src, value;
            bind.postln;
            #cc, label, src = bind;
            value = ~setA[label];
            [label, cc, value].postln;
            if( value.notNil ) {
                mout.control(0, cc, (value * 127.0).asInteger);
            };
        };

    };

    ~asynthdef = {|fback=0.8, sourcelisten=0.2, sourcelisten_gate=1, micgain=1, locut=50, hicut=16000, revwet=0, eq=1, tnoisesamp=0, clean=0.8, attendb=18, noisesmooth=0.95, testnoise=0, balratio=0.9, balwet=1, a1=0, a2=0, a3=0, distortion=0, mica2feed=0, mica2listen=0.5|
        // ins and outs
        var fb0 = FbNode(1, 0.2);
        var bus_listen     = 0;          // this receives a stereo signal, so 0, 1
        var bus_headphone1 = 2;
        var bus_headphone2 = 3;
        var sourceA = SoundIn.ar(0);
        var sourceB = SoundIn.ar(1);
        var mic_a2  = SoundIn.ar(2);
        var mic_b4  = SoundIn.ar(3);
        var mic_b3  = SoundIn.ar(4);
        var extragain = 1 + (a1*\a1mul.kr(0)) + (a2*\a2mul.kr(0.85)) + (a3*\a3mul.kr(0)) lag: 0.1;
        var pitchshift_curve = Env.bpflinear(
            -40.dbamp, 0,
            -30.dbamp, 0,
            -18.dbamp, 0.1,
            -6.dbamp, 0.5,
            0.dbamp, 1
        );
        var pitchshift_wet_curve = Env.bpflinear(
            -40.dbamp, 0,
            -25.dbamp, 0,
            -20.dbamp, 0.5,
            -10.dbamp, 1
        );
        var osctrig = Impulse.kr(5);
        var distvalue = distortion * \distmax.kr(1);

        // the pitch-shifting contraption
        var pitchshift_anal   = Amplitude.kr(mic_a2, 0.1, 0.1).lag(0.1);
        var pitchshift_freq   = IEnvGen.kr(pitchshift_curve, pitchshift_anal) * \shiftscale.kr(12) lag: 0.1 !>_.sendosc(label:"pshiftfreq", trig:osctrig);
        var pitchshift_relwet = IEnvGen.kr(pitchshift_wet_curve, pitchshift_anal).sendosc(label:"pshiftrelwet", trig:osctrig);

        var sourcedist = {|in, amount|
            in.oversamp2 {|in|
                EM_CubicDistortion.ar(in, amount, amount*0.7) >< LOSER_WaveShapeDist2.ar(in, amount.lag(0.15)) @0.5 !> LeakDC.ar(_);
                // FaustCubicDistortion.ar(in, amount, amount*0.7) >< LOSER_WaveShapeDist2.ar(in, amount.lag(0.15)) @0.5;
            };
        };

        var sourceA_post = (
            sourceA + (mic_a2 * mica2feed)
            /*
            !>< {|in|
                in >< (in !> sourcedist.(_, distvalue.lag(0.2))) @((distvalue > 0.1).lag(0.1))
                !>< Balance.ar(_, sourceA) @\distbal.kr(0)
            } @\distmix.kr(0)
            */
        );

        //var tmp = mic_a2 * mica2listen !> Amplitude.kr(_) !>_.sendosc(path:"/print/vu", label:"mica2", trig:osctrig);

        var sourceB_post = WhiteNoise.ar(0.2);
        var freqshift = {|in|
            LPF.ar(in, 6000)
            !> ~dcshift.(_, pitchshift_freq)
        };

        var a2listen = (mic_a2 * mica2listen !>< freqshift.(_, pitchshift_freq) @(pitchshift_relwet * \pshiftwet.kr(0) * 0.5));
        // var a2listen = (mic_a2 * mica2listen);
        var is_b3_down = EM_Hysteresis.kr(mic_b3, thresh_att: -30.dbamp, thresh_rel: -50.dbamp, attack:0.08, hold:0.08, release:0.1) !>_.sendosc(label:"b3-down", trig:osctrig);

        var micb3post = (
            mic_b3 !> DelayConst.ar(_, ~conf.micb3_fftsize/SampleRate.ir)
            >< PartConv.ar(mic_b3* \b3convgain.kr(1), ~conf.micb3_fftsize, ~conf.micb3_buf.value) @\b3convwet.kr(0)
            !>< GVerb.ar(_, roomsize:4, revtime:3, damping:0.3) @\b3revwet.kr(0)
        );

        var delayed = (
            mic_b4 + mic_b3 +  (micb3post[0] * \b3feed.kr(1))
            // !> BHiCut.ar(_, hicut, order:3)
            !> BLowCut.ar(_, locut, order:3)
            * (extragain * micgain)
            !>< Balance.ar(_, Amplitude.ar(fb0, 0.01, 0.2) * balratio) @balwet
            !>< freqshift.(_) @(pitchshift_relwet * \pshiftwet.kr(0))
        );

        // FDN DISABLED
        /*
        var combdelay = a1 * 0.003 + (a2*0.006) + (a3*0.012) !>_.lag(0.1);
        var fdn = 7.collect {|i|
            var ratio = i.linlin(0, 7, 0.1, 0.9);
            var dt = combdelay * ratio + (ratio*0.1);
            var decayi = Rand(0.5, 2) * fback;

            FaustSDelay.ar(delayed, dt, Elib.decay2fback(3 * fback, dt))

        } !> Mix(_) * \fdngain.kr(0.001);
        */
        var fdn = Silence.ar;

        var sourcelisten_gain = (sourcelisten * sourcelisten_gate).lag(0.4);
        var ff = (
            sourceA_post * sourcelisten_gain
            // + (mic_a2 * mica2listen !>< FreqShift.ar(_, pitchshift_freq) @(pitchshift_relwet * \pshiftwet.kr(0) * 0.5))
            + a2listen
            + (micb3post * \b3listen.kr(1))
            + (fdn * \fdnlisten.kr(1));
        );
        var tolisten_mono = (
            (delayed * (1 - sourcelisten_gain))
            + ff
        );
        var tolisten_stereo = [ tolisten_mono, DelayConst.ar(delayed * (-1), 0.1/343) + ff ];

        var tolisten = (
            tolisten_stereo
            >< GVerb.ar(tolisten_mono, roomsize:2, revtime:1) @revwet
            !> EM_Limit(_)
        );  // tolisten should be a stereo stream

        sourceA_post + (fb0 * fback) * \phonegain1.kr(1) !> Out(bus_headphone1);
        sourceB_post + (fb0 * fback) * \phonegain2.kr(1) !> Out(bus_headphone2);

        delayed !> fb0.write(_);
        tolisten * \mastergain.kr(1) !> Out(bus_listen);
        //pitchshift_anal.ampdb.sendosc(label:"pshiftdb_", trig:osctrig);

        0;

    };

    // -- - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - -- - - - --

    a = ~asynthdef.play;
    1.5.wait;
    synthargs = a.synthDef.allControlNames.collect (_.name);
    a.setn(* ~setA.select{|v, k| synthargs.indexOf(k).notNil}.asArgsArray);

    s.sync;  // ..............................................................
    fbackcurve = Env.bpflinear(0, 0, 0.1, 0, 0.2, 0.5, 0.78, 0.5, 0.97, 1, 1, 1);

    fbackset = {|rel, lo, hi, eq|
        var fback;

        rel = rel ? ~setA.fbackrel ? 0;
        lo = lo ? ~setA.fbacklo ? 0;
        hi = hi ? ~setA.fbackhi ? 1;
        eq = eq ? ~setA.fbackeq ? 0.8;
        if( rel < 0.5 ) {
            fback = fbackcurve.at(rel).linlin(0, 0.5, lo, eq);
        } {
            fback = fbackcurve.at(rel).linlin(0.5, 1, eq, hi);
        };
        ~setA.fback = fback;
        ~setA.fbacklo = lo;
        ~setA.fbackhi = hi;
        ~setA.fbackrel = rel;
        ~setA.fbackeq = eq;
        a.set(\fback, fback);
        "fback: %".format(fback).postln;
        NetAddr("localhost", 31415)
        .sendMsg("/print", "fback", fback)
        .sendMsg("/print", "fback lo", lo)
        .sendMsg("/print", "fback hi", hi)
        .sendMsg("/print", "fback eq", eq)

    };

    setvalue = {|label, value|
        a.set(label, value);
        ~setA[label] = value;
        printaddr.sendMsg("/print", label, value);
    };

    ccbind = {|cc, label, func|
        a.ccbind((cc:cc, src:MIDIClient.getSource("BCF")), label, func, store:~setA, post:31415, postpath:"/print");
        if( func.isNil ) {
            ~ccbinds.add([cc, label, "BCF"]);
        };
    };

    //ccbind.(BCF.fader_cc(1), \fback, _/127*2);
    // MIDIFunc.cc( ccNum:BCF.fader_cc(2), func: {|x| fbackset.(rel: x/127)} );
    ccbind.(BCF.fader_cc(1), \sourcelisten);
    ccbind.(BCF.fader_cc(2), \mica2feed);
    ccbind.(BCF.fader_cc(3), \mica2listen);
    ccbind.(BCF.fader_cc(4), \b3convgain);
    ccbind.(BCF.fader_cc(5), \b3convwet);

    ccbind.(BCF.fader_cc(6), \b3revwet);
    ccbind.(BCF.fader_cc(7), \b3feed);
    ccbind.(BCF.fader_cc(8), \b3listen);


    // ccbind.(BCF.fader_cc(7), \balratio);
    //ccbind.(BCF.fader_cc(8), \balwet);

    ccbind.(BCF.knob_cc(1, 2), \locut,   _.linexp(0, 127, 20, 3000));
    ccbind.(BCF.knob_cc(2, 2), \hicut,   _.linexp(0, 127, 400, 16000));
    ccbind.(BCF.button_cc(1, 2), \a3, _>0*1);
    ccbind.(BCF.button_cc(2, 2), \a2, _>0*1);
    ccbind.(BCF.button_cc(2, 1), \a2, _>0*1);
    ccbind.(BCF.button_cc(3, 2), \a1, _>0*1);

    ccbind.(BCF.knob_cc(1, 3), \a3mul);
    ccbind.(BCF.knob_cc(2, 3), \a2mul);
    ccbind.(BCF.knob_cc(3, 3), \a3mul);
    ccbind.(BCF.knob_cc(3, 1), \micgain);
    ccbind.(BCF.knob_cc(4, 3), \revwet);
    ccbind.(BCF.knob_cc(5, 3), \distmax);
    ccbind.(BCF.knob_cc(6, 3), \distmix);
    ccbind.(BCF.knob_cc(7, 3), \distbal);
    ccbind.(BCF.knob_cc(8, 3), \pshiftwet);
    ccbind.(BCF.knob_cc(1, 4), \mastergain);
    ccbind.(BCF.knob_cc(2, 4), \fdngain);
    ccbind.(BCF.knob_cc(3, 4), \fdnlisten);

    MIDIFunc.cc(ccNum:BCF.knob_cc(3, 2), func: {|x| fbackset.(lo: (x/127*2).postpp("fbacklo: ")) });
    MIDIFunc.cc(ccNum:BCF.knob_cc(4, 2), func: {|x| fbackset.(hi: (x/127*2).postpp("fbackhi: ")) });
    MIDIFunc.cc(ccNum:BCF.knob_cc(5, 2), func: {|x| fbackset.(eq: (x/127*2).postpp("fbackeq: ")) });
    ccbind(a, BCF.knob_cc(6, 2), \phonegain1);
    ccbind(a, BCF.knob_cc(7, 2), \phonegain2);

    MIDIFunc.cc( {|x| fbackset.(rel: x/127)}, ccNum: 121, chan:0);   // P1 --> BCF

    ~bcf_switch = (cc: 93, chan: 0);
    ccbind.(~bcf_switch, \sourcelisten_gate);

    distortion_curve = Env.bpflinear(0, 0, 0.3, 0, 0.6, 0.8, 0.9, 0.92);
    /*
    OSCFunc({|msg|
        var x = msg[1];
        var x2 = distortion_curve.at(x);
        printaddr.sendMsg("/print", "distortion", x2);
        a.set(\distortion, x2);
    }, path:"/firmata/a0");
    */
    OSCFunc({|msg|
        var x = msg[1];
        var x2 = distortion_curve.at(x);
        printaddr.sendMsg("/print", "sourcelisten", x2);
        a.set(\sourcelisten, x2);
    }, path:"/firmata/a0");

    /*
    ccstore = {|label, func, cc, chan, src, changed, obj=~setA|
        var with_changed, no_changed, myfunc;
        if( func.isNil ) {
            func = {|val| val/127 };
        };
        with_changed = {|value|
            value = func.(value);
            changed.(value);
            obj[label] = value;
        };
        no_changed = {|value|
            value = func.(value);
            obj[label] = value;
        };

        if(changed.notNil) {
            myfunc = with_changed;
        }{
            myfunc = no_changed;
        };
        ^MIDIFunc.cc(ccNum:cc, chan:chan, src:src, func:myfunc);
    };
    */

    // -----------------------------------------------------------------------

    ~persist_func = {
        ~savesettings.(~setA, ~setApath);
    };

    ~persist = Routine {
        loop {
            ~persist_func.();
            10.0.wait;
        }
    }.play(AppClock);

    CmdPeriod.doOnce({
        {
            ~persist_func.();
        }.defer;
    });

    {
        var in = SoundIn(3) + SoundIn(4) * 0.5;
        var hasf, freq;
        #freq, hasf = Tartini.kr(in);
        freq = freq.lag(0.4);
        freq.cpsmidi.sendosc(label:"detectednote_", trig:Impulse.kr(4));
        freq.floor.sendosc(label:"detectedfreq_", trig:Impulse.kr(4));

        // freq.poll(4);
    }.play;

    ">>>>>>>>>>>>>>>>>".postln;
    ~ccbinds_reflect.();



}
)

// ______________________________________________________________

// test headphones
(
{
    SinOsc.ar(440) !> Out(2);
    WhiteNoise.ar !> Out(3);
}.play;
)


{ SoundIn(3) !> Out(0) }.play
{ WhiteNoise.ar(0.1) !> Out([0, 1]) }.play
{ SoundIn(4) !> Out(0) }.play
{ SinOsc.ar(440) !> Out(0) }.play
{ SoundIn.ar(2) !> Out(0) }.play
{ SinOsc.ar(440) ++ PinkNoise.ar(0.2) !> Out(8) }.play



Server

~eq = MasterEQ()
