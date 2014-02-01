/*
README
======

This is the main patch for __________, by Eduardo Moguillansky

1) Go down to SETUP and modify as indicated to fit your setup
2) There are two code-blocks to evaluate to get the patch started
   You evaluate a code-block by placing your cursor at the _____
   in each "EVAL THIS" sign, and pressing Cmd-ENTER.
   Eval each of the code-blocks


===================================================
=                                                 =
=                     SETUP                       =
=                                                 =
===================================================

*/

(
///////////////////////////////////////////////////////////////////////
//                     EVAL THIS: ________________
///////////////////////////////////////////////////////////////////////

/// ---------------- CONFIGURATION ----------------

// Base path of the project. Change it when moving it to another place.
// This should reflect the path where this file was saved
~basepath = "~/proj/grame".absolutePath;   // <------------------- CHANGE THIS WHEN MOVING TO ANOTHER PLACE!!

// CHANNELS begin with 1
~input_channels = (
    \wg1_A2 : 3,
    \wg1_B4 : 4,
    \wg1_B3 : 5,
    \wg2_A2 : 6,
    \wg2_B3 : 7,
    \wg2_B4 : 8,
    \busA   : 1,
    \busB   : 2
);

~output_channels = (
    \wg1_speakerL : 3,
    \wg1_speakerR : 4,
    \wg2_speakerL : 5,
    \wg2_speakerR : 6,
    \wg1_listen   : 1,
    \wg2_listen   : 2
);

// ------------------ END CONFIGURATION --------------------
~input_channels.keysValuesChange {|k, v| v-1};
~output_channels.keysValuesChange {|k, v| v-1};

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

~dumpsettings = {|settings|
    var keys = settings.keys.asArray;
    keys.sort.do {|k|
        "% : %".format(k, settings[k]).postln;
    };
};


~setApath = ~basepath +/+ "CONFIG" +/+ "setA.txt";

~conf = ();
~conf.micb3_fftsize = 2048;
~conf.micb3_bufpath = ~basepath +/+ "ASSETS" +/+ "A4174cm.wav";
if( ~conf.micb3_buf.value.notNil ) {
    "Freeing previosly allocated Buffer: %".format(~conf.micb3_buf.value).postln;
    ~conf.micb3_buf.value.free;
    ~conf.micb3_buf.value = nil;
};
~conf.micb3_buf = PartConv.getBuffer(~conf.micb3_bufpath, ~conf.micb3_fftsize);

Elib.midi_init;
)

(
///////////////////////////////////////////////////////////////////////
//                     EVAL THIS: ________________
///////////////////////////////////////////////////////////////////////
fork {
    var synthargs, fbackset, fbackcurve, pedalcurve, ccbind, setvalue, checkNil, bcfbind, synth1, synth2;
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

    /*
    -------------------------------------------------
    -
    -  SYNTHESIS
    -
    -------------------------------------------------
    */

    checkNil = {|in, msg|
        if(in.isNil) {
            "################# NIL found! #################".postln;
            msg.throw;
        };
        in;
    };

    // SET THE CHANNELS HERE
    SynthDef(\wg, {|in_A2, in_B3, in_B4, in_busA, in_busB, out_listen, out_speakerL, out_speakerR,
                   fback=0.8, sourcelisten=0.2, sourcelisten_gate=1, micgain=1, locut=50, hicut=16000, revwet=0, eq=1,
                   balratio=0.9, balwet=1,
                   distortion=0, mica2feed=0, mica2listen=0.5|
        // ins and outs
        var fb0 = FbNode(1, 0.2);

        // ----------------------------------
        //
        // SET IO HERE!!
        //
        // ----------------------------------

        // OUTPUTS
        var bus_listen     = out_listen;          // this receives a stereo signal, so 0, 1
        var bus_headphone1 = out_speakerL;          // This needs to match the routing to the headphone-amplifier 1=L, 2=R
        var bus_headphone2 = out_speakerR;

        // INPUTS
        var sourceA = SoundIn.ar(in_busA);
        var sourceB = SoundIn.ar(in_busB);
        var mic_a2  = SoundIn.ar(in_A2);
        var mic_b4  = SoundIn.ar(in_B4);
        var mic_b3  = SoundIn.ar(in_B3);

        var pitchshift_curve = Env.bpflinear(
            -50.dbamp, 0,
            -40.dbamp, 0,
            -24.dbamp, 0.1,
            -18.dbamp, 0.5,
            0.dbamp, 1
        );

        var pitchshift_wet_curve = Env.bpflinear(
            -40.dbamp, 0,
            -28.dbamp, 0,
            -20.dbamp, 0.5,
            -10.dbamp, 1
        );

        var osctrig = Impulse.kr(5);
        var a2_amp = Amplitude.kr(mic_a2, 0.1, 0.1).lag(0.1) !> _.oscprint(label:"A2-amp_", trig:osctrig, convfunc:(_.ampdb));

        /* DISTORTION DISABLED
        var distvalue = distortion * \distmax.kr(1);
        var sourcedist = {|in, amount|
            in.oversamp2 {|in|
                EM_CubicDistortion.ar(in, amount, amount*0.7) >< LOSER_WaveShapeDist2.ar(in, amount.lag(0.15)) @0.5 !> LeakDC.ar(_);
            };
        };
        */

        var mica2_feed   = mic_a2*mica2feed;
        var sourceA_post = sourceA + mica2_feed;
        var sourceB_post = sourceB + mica2_feed;
        var sources      = sourceA_post+sourceB_post*0.707;
        var sources_amp = Amplitude.kr(sources);

        var a2_ratio   = a2_amp/sources_amp !> _.oscprint("a2ratio_", trig:osctrig);
        var a2_pressed = a2_ratio.clip(0, 1);
        var pitchshift_freq = a2_pressed * \shiftscale.kr(12) lag:0.1 !>_.oscprint(label:"A2-doppler_", trig:osctrig);
        var pitchshift_relwet = a2_pressed;

        var freqshift = {|in, freq|
            LPF.ar(in, 6000)
            !> FreqShift.ar(_, freq);
        };

        var a2listen = (mic_a2 * mica2listen) !>< freqshift.(_, pitchshift_freq) @ (pitchshift_relwet * \shiftwet.kr(1) );

        var b3_amp = Amplitude.kr(mic_b3, 0.1, 0.1) !> _.oscprint(label:"B3_amp_", trig:osctrig, convfunc:(_.ampdb));

        // var b3_ratio = RunningSum.rms_kr(mic_b3)/RunningSum.rms_kr(sources, 100) * (a2_pressed*7+1) !> _.oscprint("b3ratio_", trig:osctrig);
        var b3_ratio = Amplitude.kr(mic_b3) / sources_amp * (a2_pressed*7+1) !> _.oscprint("b3ratio_", trig:osctrig);
        var b3_pressed = (
            Env.bpflinear(
                0, 0,
                0.03, 0,
                0.07, 0.5,
                0.12, 1
            )
            !> IEnvGen.kr(_, b3_ratio)
            !> _.oscprint(label:"B3-pressed", trig:osctrig)
        );

        var micb3env = Amplitude.kr(sources, attackTime:0.001, releaseTime:2*fback);

        var micb3_reson = mic_b3 !><.0 GVerb.ar(_, roomsize:4, revtime:3, damping:0.3) @\b3revwet.kr(0);
        var micb3_conv = (
            mic_b3 * (\b3convgain.kr(1) * LinSelectX.kr(\b3follow.kr(0), 1, micb3env))
            !>< PartConv.ar(_, ~conf.micb3_fftsize, ~conf.micb3_buf.value) @\b3convwet.kr(0)
            !><.0 GVerb.ar(_, roomsize:4, revtime:3, damping:0.3) @\b3revwet.kr(0);
        );
        var micb3post = mic_b3 >< (micb3_conv) @ fback.linlin(0, 1, 0.5, 1);
        /*
        var micb3post2 = (
            mic_b3
            * (\b3convgain.kr(1) * LinSelectX.kr(\b3follow.kr(0), 1, micb3env))
            !><.0 GVerb.ar(_, roomsize:4, revtime:3, damping:0.3) @\b3revwet.kr(0) // GVerb generates a stereo output
            // !> DelayConst.ar(_, ~conf.micb3_fftsize/SampleRate.ir)
            !>< PartConv.ar(_, ~conf.micb3_fftsize, ~conf.micb3_buf.value) @\b3convwet.kr(0)

            //!>< (_*micb3env) @\b3follow.kr(0)
        ) >< mic_b3 @ (1-fback2revwet);
        */
        var delayed = (
            mic_b4
            + (mic_b3 >< micb3post @ \b3wet.kr(0))
            // * micgain
        );

        /*
        var combdelay = (a2_pressed>0.5) * 0.003 + (b3_pressed>0.5*0.01) !>_.lag(0.01) !> _.oscprint("fdn::d_", trig:osctrig);
        var fdn = 7.collect {|i|
            var ratio = i.linlin(0, 7, 0.1, 0.9);
            var dt = combdelay * ratio + (ratio*0.1);
            var decayi = Rand(0.5, 2) * fback;
            FaustSDelay.ar(delayed, dt, Elib.decay2fback(3 * fback, dt))
        } !> Mix(_) * \fdngain.kr(0.001);
        */

        var ff = (
            sourceA_post * sourcelisten
            // + (mic_a2 * mica2listen !>< FreqShift.ar(_, pitchshift_freq) @(pitchshift_relwet * \pshiftwet.kr(0) * 0.5))
            + a2listen
            + (micb3post * \b3listen.kr(1))
            // + (fdn * \fdnlisten.kr(1));
        );
        var tolisten_mono = (
            delayed >< ff @ \ffwet.kr(0);
        );

        /*
        var tolisten_stereo = [ tolisten_mono, DelayConst.ar(delayed * (-1), 0.1/343) + ff ];
        var tolisten = (
            tolisten_stereo
            >< GVerb.ar(tolisten_mono, roomsize:2, revtime:1) @revwet
            !> EM_Limit(_, ratio:5, release:0.05)
        );  // tolisten should be a stereo stream
        */
        var tolisten = (
            tolisten_mono
            >< GVerb.ar(tolisten_mono, roomsize:2, revtime:1)[0] @revwet
            !> EM_Limit(_, ratio:5, release:0.05)
        );  // MONO

        Amplitude.kr(delayed).lagud(0.1, 0.3).oscsend(31415, "/print/vu", "_delayed", trig:osctrig);
        Amplitude.kr(ff).lagud(0.1, 0.3).oscsend(31415, "/print/vu", "_ff", trig:osctrig);

        sourceA_post + (fb0 * fback) * \speakergainL.kr(1) !> Out(bus_headphone1);
        sourceB_post + (fb0 * fback) * \speakergainR.kr(1) !> Out(bus_headphone2);

        fb0.write(delayed);
        tolisten * \mastergain.kr(1) !> Out(bus_listen);

    }).add;

    s.sync;

    // -- - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - -- - - - --

    synth1 = Synth(\wg, args:[
        in_A2:   ~input_channels[\wg1_A2],
        in_B3:   ~input_channels[\wg1_B3],
        in_B4:   ~input_channels[\wg1_B4],
        in_busA: ~input_channels[\busA],
        in_busB: ~input_channels[\busB],
        out_listen:   ~output_channels[\wg1_listen],
        out_speakerL: ~output_channels[\wg1_speakerL],
        out_speakerR: ~output_channels[\wg1_speakerR]
    ]);

    ~bcf = MIDIClient.getSource("BCF");
    if (~bcf.isNil ) {
        "BCF2000 not found".throw;
    };

    1.5.wait;
    synthargs = synth1.synthDef.allControlNames.collect (_.name);
    synth1.setn(* ~setA.select{|v, k| synthargs.indexOf(k).notNil}.asArgsArray);

    s.sync;  // ..............................................................
    fbackcurve = Env.bpflinear(0, 0, 0.08, 0, 0.15, 0.5, 0.78, 0.5, 0.97, 1, 1, 1);

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
        synth1.set(\fback, fback);
        NetAddr("localhost", 31415)
        .sendMsg("/print", "fback", fback)
        .sendMsg("/print", "fback lo", lo)
        .sendMsg("/print", "fback hi", hi)
        .sendMsg("/print", "fback eq", eq)
    };

    setvalue = {|label, value|
        synth1.set(label, value);
        ~setA[label] = value;
        printaddr.sendMsg("/print", label, value);
    };

    ccbind = {|cc, label, func|
        synth1.ccbind((cc:cc, src:~bcf), label, func, store:~setA, post:31415, postpath:"/print");

        if( func.isNil ) {
            ~ccbinds.add([cc, label, "BCF"]);
        };
    };

    bcfbind = {|nodename, kind, which, param, func|
        var label;
        var synth = if( nodename == \wg1 ) {synth1} {synth2};

        if( kind == \fader ) {
            label = param++" (F"++which++")";
            MIDIFunc.ccbind((cc:BCF.fader_cc(which), src:~bcf), a, param, spec:func, post:(path:"/print", port:31415, label:label), store:~setA);
        };
        if( kind == \knob ) {
            label = "% (K%,%)".format(param, which[0], which[1]);
            MIDIFunc.ccbind((cc:BCF.knob_cc(which[0], which[1]), src:~bcf), a, param, spec:func, post:(path:"/print", port:31415, label:label), store:~setA);
        };
        label.postln;
    };

    // ------------- BINDINGS -----------------------------------------

    bcfbind.(\wg1, \fader, 1, \ffwet);
    bcfbind.(\wg1, \fader, 2, \mica2feed);
    bcfbind.(\wg1, \fader, 3, \mica2listen);
    bcfbind.(\wg1, \fader, 4, \b3convgain);
    bcfbind.(\wg1, \fader, 5, \b3convwet);
    bcfbind.(\wg1, \fader, 6, \b3revwet);
    bcfbind.(\wg1, \fader, 7, \b3wet);
    bcfbind.(\wg1, \fader, 8, \b3listen);

    bcfbind.(\wg1, \knob, [1, 2], \locut, _.linexp(0, 127, 20, 3000));
    bcfbind.(\wg1, \knob, [2, 2], \hicut, _.linexp(0, 127, 400, 16000));
    MIDIFunc.cc(ccNum:BCF.knob_cc(3, 2), func: {|x| fbackset.(lo: (x/127*2)) });
    MIDIFunc.cc(ccNum:BCF.knob_cc(4, 2), func: {|x| fbackset.(eq: (x/127*2)) });
    MIDIFunc.cc(ccNum:BCF.knob_cc(5, 2), func: {|x| fbackset.(hi: (x/127*2)) });
    MIDIFunc.cc(ccNum:BCF.knob_cc(6, 2), func: {|x| fbackset.(rel: x/127)} );
    MIDIFunc.cc(ccNum: 65,               func: {|x| fbackset.(rel: x/127)} );
    bcfbind.(\wg1, \knob, [7, 2], \speakergainL);
    bcfbind.(\wg1, \knob, [8, 2], \speakergainR);

    bcfbind.(\wg1, \knob, [3, 3], \revwet);
    bcfbind.(\wg1, \knob, [7, 3], \shiftscale, _/127*24);
    bcfbind.(\wg1, \knob, [8, 3], \shiftwet);

    bcfbind.(\wg1, \knob, [1, 4], \mastergain);
    bcfbind.(\wg1, \knob, [2, 4], \sourcelisten);
    bcfbind.(\wg1, \knob, [3, 4], \fdngain);
    bcfbind.(\wg1, \knob, [4, 4], \fdnlisten);
    bcfbind.(\wg1, \knob, [5, 4], \b3follow);

    ~bcf_switch = (cc: 93, chan: 0);
    ccbind.(~bcf_switch, \sourcelisten_gate);

    fbackset.(lo: ~setA.fbacklo);
    fbackset.(hi: ~setA.fbackhi);
    fbackset.(eq: ~setA.fbackeq);
    fbackset.(rel: ~setA.fbackrel);


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

    ~detect_note = {
        var in = SoundIn(3) + SoundIn(4) * 0.5;
        var hasf, freq;
        #freq, hasf = Tartini.kr(in);
        freq = freq.lag(0.4);
        freq.cpsmidi.oscprint(label:"detectednote_", trig:Impulse.kr(4));
        freq.floor.oscprint(label:"detectedfreq_", trig:Impulse.kr(4));
    }.play;

    ">>>>>>>>>>>>>>>>>".postln;
    ~ccbinds_reflect.();

    1.0.wait;
    ~dumpsettings.(~setA);

}
)

// ______________________________________________________________
~dumpsettings.(~setA)

(
~live = {
    SoundIn.ar(10) !> Out.ar(0, _);
}.play
)

b.free