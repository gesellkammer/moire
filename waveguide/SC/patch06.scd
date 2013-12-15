/*
README
======

This is the main patch for "moire", by Eduardo Moguillansky

1) Go down to SETUP and modify as indicated to fit your setup
2) There are two code-blocks to evaluate to get the patch started
   You evaluate a code-block by placing your cursor at the _____
   in each "EVAL THIS" sign, and pressing Cmd-ENTER.
   Eval each of the code-blocks
3) After evaluating the second


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
fork {
    ~basepath = "~/proj/moire/waveguide/SC".absolutePath;   // <------------------- CHANGE THIS WHEN MOVING TO ANOTHER PLACE!!

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

    ~gui_oscport = 8801;

    // ------------------ END CONFIGURATION --------------------
    ~input_channels.keysValuesChange {|k, v| v-1};
    ~output_channels.keysValuesChange {|k, v| v-1};

    ~readsettings = {|path|
        var out;
        path = path.absolutePath;
        "reading settings from %".format(path).postln;
        out = if(File.exists(path)) {
            var file, str, dict;
            file = File(path, "r");
            str = file.readAllString;
            "settings: %".format(str).postln;
            dict = str.interpret;
            if ( dict.isKindOf(Dictionary).not ) {
                "settings: problem reading the saved settings".postln;
                ();
            } {
                dict;
            };
        }{
            ();  // <---- defaults to an empty Event
        };
        "finished reading settings".postln;
        out;
    };

    ~savesettings = {|dict, path|
        var file = File(path.absolutePath, "w");
        var str = dict.asCompileString;
        file.write(str);
        file.close;
    };

    ~dumpsettings = {|settings|
        var keys = settings.keys.asArray;
        keys.sort.do {|k|
            "% : %".format(k, settings[k]).postln;
        };
    };

    ~setpaths = [~basepath +/+ "CONFIG" +/+ "setA.txt", ~basepath +/+ "CONFIG" +/+ "setB.txt"];
    ~b3_fftsize = 2048;
    ~b3_bufpaths = [~basepath +/+ "ASSETS" +/+ "A4174cm.wav", ~basepath +/+ "ASSETS" +/+ "A4174cm.wav"];
    if( ~b3_bufs.notNil ) {
        ~b3_bufs.do {|bufref|
            if(bufref.notNil && bufref.value.notNil) {
                "Freeing previosly allocated Buffer: %".format(bufref.value).postln;
                bufref.value.free;
                bufref.value = nil;
            };
        };
    };
    ~b3_bufs = ~b3_bufpaths.collect {|path| PartConv.getBuffer(path, ~b3_fftsize) };

    Elib.midi_init;
};
)

(
///////////////////////////////////////////////////////////////////////
//                     EVAL THIS: ________________
///////////////////////////////////////////////////////////////////////
fork {
    var synthargs, fbackset, fbackcurve, pedalcurve, ccbind, checkNil, bcfbind, synth1, synth2;
    //var guiaddr = NetAddr("localhost", 31415);
    var guiaddr   = NetAddr("localhost", ~gui_oscport);

    Elib.midi_init;

    if( ~conf.isNil ) {
        "\n>>>>>>>>>>> evaluate SETUP first! <<<<<<<<<<<<<\n".throw;
    };

    if ( ~b3_bufs[0].value.isNil ) {

        "+--------------------------+".postln;
        "| WARNING: buffer not set! |".postln;
        "+--------------------------+".postln;

    };

    ~setA = ~readsettings.(~setpaths[0]).postln;
    ~setB = ~readsettings.(~setpaths[1]).postln;
    ~sets = [~setA, ~setB];

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
    SynthDef(\wg, {|i_id, in_A2, in_B3, in_B4, in_busA, in_busB, out_listen, out_speakerL, out_speakerR, b3bufnum,
                   fbackrel=0.5, fbacklo=0.2, fbackhi=1.2, fbackeq=0.91,
                   sourcelisten=0.2, locut=50, hicut=16000, revwet=0, mica2feed=0, mica2listen=0.5|
        // ins and outs
        var fb0 = FbNode(1, 0.2);

        // FEEDBACK
        var fbackwarp = (
            Env.bpflinear(0, 0, 0.08, 0, 0.15, 0.5, 0.78, 0.5, 0.97, 1, 1, 1)
            !> IEnvGen.kr(_, fbackrel)
        );
        var fback = (
            Select.kr( fbackwarp >= 0.5, [fbackwarp.linlin(0, 0.5, fbacklo, fbackeq), fbackwarp.linlin(0.5, 1, fbackeq, fbackhi)] )
        );
        var fbackshow = fback.oscprint("fback", Changed.kr(fback));

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

        var guiport = ~gui_oscport;

        var osctrig = Impulse.kr(10);
        var a2_amp = Amplitude.kr(mic_a2, 0.1, 0.1).lag(0.1);

        var mica2_feed   = mic_a2*mica2feed;
        var sourceA_post = sourceA + mica2_feed;
        var sourceB_post = sourceB + mica2_feed;
        var sources      = sourceA_post+sourceB_post*0.707;
        var sources_amp = Amplitude.kr(sources, attackTime:0.005, releaseTime:0.02);

        var a2_ratio   = a2_amp/sources_amp.max(0.000001) !> _.oscprint("a2ratio_", trig:osctrig);

        var a2_pressed = (
            (sources_amp > -50.dbamp) * a2_ratio
            !> _.clip(0, 1)
            !> _.lag2(0.1)
            !> _.oscprint(label:"A2-pressed_", trig:osctrig)
        );

        var pitchshift_freq = a2_pressed * \shiftscale.kr(12) lag:0.1 !> _.oscprint(label:"A2-doppler_", trig:osctrig);

        var pitchshift_relwet = a2_pressed;

        var freqshift = {|in, freq|
            LPF.ar(in, 6000)
            !> FreqShift.ar(_, freq);
        };

        var dopplerwet = (pitchshift_relwet * \shiftwet.kr(1));

        var a2listen = (mic_a2 * mica2listen)  !>< freqshift.(_, pitchshift_freq) @ dopplerwet;

        var b3_amp = Amplitude.kr(mic_b3, 0.1, 0.1).max(0.0000000001) !> _.oscprint(label:"B3_amp_", trig:osctrig, convfunc:(_.ampdb));

        var b3_ratio = (b3_amp/sources_amp) * a2_pressed.linlin(0, 1, 1, 8) !> _.oscprint("b3ratio_", trig:osctrig);
        var b3_pressed = (
            Env.bpflinear(
                0, 0,
                0.03, 0,
                0.07, 0.5,
                0.12, 1
            )
            !> IEnvGen.kr(_, b3_ratio)
            * (b3_amp > 0.000001)
            !> _.oscprint(label:"B3-pressed_", trig:osctrig)
        );

        var micb3_env   = Amplitude.kr(sources, attackTime:0.001, releaseTime:2*fback);
        var micb3_reson = mic_b3 !><.0 GVerb.ar(_, roomsize:4, revtime:3, damping:0.3) @\b3revwet.kr(0);
        var micb3_conv  = (
            mic_b3 * (\b3convgain.kr(1) * LinSelectX.kr(\b3follow.kr(0), 1, micb3_env))
            //+ WhiteNoise.ar(-90.dbamp) // denormals?
            !>< PartConv.ar(_, ~b3_fftsize, b3bufnum) @\b3convwet.kr(0)
            !><.0 GVerb.ar(_, roomsize:4, revtime:3, damping:0.3) @\b3revwet.kr(0);
        );
        var micb3post = mic_b3 >< (micb3_conv) @ fback.linlin(0, 1, 0.5, 1);

        var delayed = (
            mic_b4 // !>< freqshift.(_, pitchshift_freq) @ (dopplerwet*0.5)
            + (mic_b3 >< micb3post @ \b3wet.kr(0))
        );

        var combdelay = (a2_pressed>0.5) * 0.003 + (b3_pressed>0.5*0.01) !>_.lag(0.01) !> _.oscprint("fdn::d_", trig:osctrig);
        var fdn = 7.collect {|i|
            var ratio = i.linlin(0, 7, 0.1, 0.9);
            var dt = combdelay * ratio + (ratio*0.1);
            var decayi = Rand(0.5, 2) * fback;
            FaustSDelay.ar(delayed, dt, Elib.decay2fback(3 * fback, dt))
        } !> Mix(_) * \fdngain.kr(0.001);

        var ff = (
            sources * sourcelisten
            + (micb3post * \b3listen.kr(1))
            + (fdn * \fdnlisten.kr(1));
        );

        var ffwet = \ffwet.kr(0);

        var tolisten_mono = (
            delayed >< ff @ ffwet
            + a2listen
        );

        // var tolisten_stereo = [ tolisten_mono, DelayConst.ar(delayed * (-1), 0.1/343) + ff ];

        var tolisten = (
            tolisten_mono
            // >< GVerb.ar(tolisten_mono, roomsize:2, revtime:1)[0] @revwet
            !> EM_Limit(_, ratio:5, release:0.05)
        );  // MONO


        /*
        var delayed_amp = Amplitude.kr(delayed).lagud(0.1, 0.3);
        var ff_amp = Amplitude.kr(ff).lagud(0.1, 0.3);
        */

        sourceA_post + (fb0 * fback) * \speakergainL.kr(1) !> Out(bus_headphone1);
        sourceB_post + (fb0 * fback) * \speakergainR.kr(1) !> Out(bus_headphone2);

        fb0.write(delayed);
        tolisten * \mastergain.kr(1) !> Out(bus_listen);

        SendReply.kr(osctrig, "/_info", [fback, pitchshift_freq, a2_amp, a2_ratio, a2_pressed], i_id);
        //SendReply.kr(osctrig, "/_vu", [delayed_amp, ff_amp], id);

    }).add;


    s.sync;

    // -- - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - -- - - - --

    "creating synths".postln;

    synth1 = Synth(\wg, args:[
        i_id: 0,
        in_A2:   ~input_channels[\wg1_A2],
        in_B3:   ~input_channels[\wg1_B3],
        in_B4:   ~input_channels[\wg1_B4],
        in_busA: ~input_channels[\busA],
        in_busB: ~input_channels[\busB],
        out_listen:   ~output_channels[\wg1_listen],
        out_speakerL: ~output_channels[\wg1_speakerL],
        out_speakerR: ~output_channels[\wg1_speakerR],
        b3bufnum: ~b3_bufs[0].value.bufnum
    ]);
    ~synths = [synth1, synth2];
    ~nodenames = [\wg1, \wg2];

    ~bcf = MIDIClient.getSource("BCF");
    if (~bcf.isNil ) {
        "BCF2000 not found!!".postln;
    };

    1.5.wait;

    ~synths.do {|node, index|
        if( node.notNil ) {
            var nodeargs = node.synthDef.allControlNames.collect (_.name);
            var nodesettings = ~sets[index];
            var used_settings = nodesettings.select{|v, k| nodeargs.indexOf(k).notNil};
            var nodename = ~nodenames[index];
            node.setn(*used_settings.asArgsArray);
            used_settings.do {|v, k|
                guiaddr.sendMsg("/%/%".format(nodename, k), v);
            };
        };
    };

    s.sync;  // ..............................................................

    bcfbind = {|nodename, kind, which, param, func|
        var label, node, chan, storedict;
        var oscpath = nodename++"/"++param;

        if( nodename == \wg1 ) {
            node = synth1;
            chan = 0;
            storedict = ~sets[0];
        } {
            node = synth2;
            chan = 1;
            storedict = ~sets[1];
        };

        if( node.notNil) {
            switch(kind,
                \fader, {
                    label = "% - % F%".format(nodename, param, which);
                    MIDIFunc.ccbind((cc:BCF.fader_cc(which), src:~bcf, chan:chan), node, param, spec:func, post:(path:"/print", port:31415, label:label), store:storedict);
                },
                \knob, {
                    label = "% - % K%/%".format(nodename, param, which[0], which[1]);
                    MIDIFunc.ccbind((cc:BCF.knob_cc(which[0], which[1]), src:~bcf, chan:chan), node, param, spec:func, post:(path:"/print", port:31415, label:label), store:storedict);
                },
                \cc, {
                    label = "% - % CC%".format(nodename, param, which);
                    MIDIFunc.ccbind((cc:which, src:~bcf, chan:chan), node, param, spec:func, post:(path:"/print", port:31415, label:label), store:storedict);
                }
            );
        } {
            "node %s is nil, skipping binding".format(nodename).postln;
        };


    };

    // ------------- GUI -----------------
    defer {
        var width = 400;
        var lineheight = 20;
        var height = 600;
        var win = Window.new(bounds:((width*2)@height)).front;
        var panels = [CompositeView(win, Rect(0, 0, width, height)), CompositeView(win, Rect(width, 0, width, height))];
        var globalpanel = CompositeView(win, Rect(0, height, width*2, height*0.25));
        var nodeset = {|index, param, value, node|
            var settings = ~sets[index];
            node = node ? ~synths[index];
            if( value.isKindOf(SimpleNumber) ) {
                settings[param] = value;
                node.set(param, value);
                "WG% : %s --> %".format(index, param, value);
            } {
                "nodeset: got %".format(value).postln;
            };
        };
        panels.do {|panel| panel.decorator = FlowLayout(panel.bounds)};
        CmdPeriod.doOnce {
            if( win.isClosed.not ) {
                win.close;
            };
        };
        ~gui = [(), ()];
        win.onClose = {
            s.freeAll;
            OSCdef.freeAll;
            MIDIdef.freeAll;

        };
        panels.do {|panel, index|
            var node = ~synths[index];
            var numberdisplay = {|panel, label, value0=0, spec|
                value0 = value0 ? 0;
                value0 = if(value0.isNaN) {0} {value0};
                EZNumber(panel, 170@lineheight, " % ".format(label), spec, labelWidth:100, initVal:value0);
            };
            var slider = {|index, param, label, spec, defaultvalue|
                var panel = panels[index];
                var settings = ~sets[index];
                var value0 = settings[param] ? defaultvalue ? 0;
                var func;
                var node = ~synths[index];
                value0 = if(value0.isNaN) {0} {value0};
                func = {|slider|
                    var v = slider.value;
                    settings[param] = v;
                    if( node.notNil ) {
                        node.set(param, v);
                    };
                };
                nodeset.(index, param, value0);
                "binding %".format(param).postln;
                EZSlider(panel, (width*0.9)@lineheight, " % ".format(label), spec, labelWidth:(width*0.4), numberWidth:60, action:func, initVal:value0);
            };
            var sliderdisplay = {|index, label, value0, spec|
                value0 = value0 ? 0;
                value0 = if(value0.isNaN) {0} {value0};
                EZSmoothSlider(panels[index], (width*0.9@lineheight), label, controlSpec:spec, labelWidth:(width*0.4), numberWidth:60);
            };

            [
                // DISPLAY
                \fback: numberdisplay.(panel, "Feedback", 0, [0, 2]),
                \dopplerfreq: numberdisplay.(panel, "Doppler Freq", 0, [0, 50]),
                \a2pressed: sliderdisplay.(index, "A2 Pressed", 0, [0, 1]),

                // SETTINGS
                \ffwet: slider.(index, \ffwet, "FF Wet", [0, 1]),
                \mica2feed : slider.(index, \mica2feed, "MIC A2 Feed", [0, 1]),
                \mica2listen : slider.(index, \mica2liste, "MIC A2 Listen", [0, 1]),
                \b3convgain : slider.(index, \b3convgain, "MIC B3 Reson Pregain", [0, 1]),
                \b3convwet  : slider.(index, \b3convwet, "MIC B3 Reson Wet", [0, 1]),
                \b3follow   : slider.(index, \b3follow, "MIC B3 Reson Follow", [0, 1], 0),
                \b3revwet   : slider.(index, \b3revwet, "MIC B3 Feedback Wet", [0, 1]),
                \b3listen   : slider.(index, \b3listen, "MIC B3 Direct Mix", [0, 1]),
                \locut  : slider.(index, \locut, "Low Cut Freq", [10, 1000], 20),
                \hicut  : slider.(index, \hicut, "High Cut Freq", [2000, 20000], 16000),
                \fbacklo  : slider.(index, \fbacklo, "Feedback Low", [0, 2], 0.2),
                \fbackeq  : slider.(index, \fbackeq, "Feedback Middle", [0, 2], 0.91),
                \fbackhi  : slider.(index, \fbackhi, "Feedback High", [0, 2], 1.3),
                \speakergainL : slider.(index, \speakergainL, "Speaker Left", [0, 1], 1),
                \speakergainR : slider.(index, \speakergainR, "Speaker Right", [0, 1], 1),
                \revwet : slider.(index, \revwet, "Global FDN Wet", [0, 1], 0),
                \shiftscale : slider.(index, \shiftscale, "Doppler Scaling", [0, 50], 12),
                \shiftwet : slider.(index, \shiftwet, "Doppler Wet", [0, 1], 0.5),
                \sourcelisten : slider.(index, \sourcelisten, "Sources Direct", [0, 1], 0.01),
                \fdngain : slider.(index, \fdngain, "Local FDN Gain", [0, 1], 0.5),
                \fdnlisten : slider.(index, \fdnlisten, "Local FDN Listen", [0, 1], 0)
            ].unlace(2).flop.do {|lwidget|
                var label = lwidget[0];
                var widget = lwidget[1];
                [label, widget].postln;
                ~gui[index].put(label, widget);
            };

        };
    };


    // ------------- BINDINGS -----------------------------------------

    ~nodenames.do {|nodename|
        "Setting synth %".format(nodename).postln;
        bcfbind.(nodename, \cc, 65, \fbackrel, _/127);

        /*

        bcfbind.(nodename, \fader, 1, \ffwet);
        bcfbind.(nodename, \fader, 2, \mica2feed);
        bcfbind.(nodename, \fader, 3, \mica2listen, _/127*2);
        bcfbind.(nodename, \fader, 4, \b3convgain);
        bcfbind.(nodename, \fader, 5, \b3convwet);
        bcfbind.(nodename, \fader, 6, \b3revwet);
        bcfbind.(nodename, \fader, 7, \b3wet);
        bcfbind.(nodename, \fader, 8, \b3listen);

        bcfbind.(nodename, \knob, [1, 2], \locut, _.linexp(0, 127, 20, 3000));
        bcfbind.(nodename, \knob, [2, 2], \hicut, _.linexp(0, 127, 400, 16000));
        bcfbind.(nodename, \knob, [3, 2], \fbacklo,  _/127*2);
        bcfbind.(nodename, \knob, [4, 2], \fbackeq,  _/127*2);
        bcfbind.(nodename, \knob, [5, 2], \fbackhi,  _/127*2);
        bcfbind.(nodename, \knob, [6, 2], \fbackrel, _/127);


        bcfbind.(nodename, \knob, [7, 2], \speakergainL, _/0);
        bcfbind.(nodename, \knob, [8, 2], \speakergainR);

        bcfbind.(nodename, \knob, [3, 3], \revwet);
        bcfbind.(nodename, \knob, [7, 3], \shiftscale, _/127*24);
        bcfbind.(nodename, \knob, [8, 3], \shiftwet);

        bcfbind.(nodename, \knob, [1, 4], \mastergain);
        bcfbind.(nodename, \knob, [2, 4], \sourcelisten);
        bcfbind.(nodename, \knob, [3, 4], \fdngain);
        bcfbind.(nodename, \knob, [4, 4], \fdnlisten);
        bcfbind.(nodename, \knob, [5, 4], \b3follow);
        */
    };

    "setting up receivers".postln;
    OSCFunc({|msg|
        var id = msg[2];
        var fback = msg[3];
        var dopplerfreq = msg[4];
        var a2_amp = msg[5];
        var a2_ratio = msg[6];
        var a2_pressed = msg[7];
        defer {
            var gui = ~gui[id];
            gui.fback.value = fback;
            gui.dopplerfreq.value = dopplerfreq;
            gui.a2pressed.value = a2_pressed;
        };
    }, path: "/_info");

    // -----------------------------------------------------------------------

    ~persist_func = {
        ~sets.do {|settings, index|
            var path = ~setpaths[index];
            ~savesettings.(settings, path);
        };
    };

    ~persist_task = Routine {
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

    ~live = {
        SoundIn.ar(10) !> Out.ar(1, _);
    }.play;

}

)

// ______________________________________________________________
~dumpsettings.(~setA)
