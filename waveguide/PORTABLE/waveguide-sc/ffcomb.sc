FaustFfcomb : UGen
{
  *ar { | in1, m(1.0), b0(1.0), bm(0.0) |
      ^this.multiNew('audio', in1, m, b0, bm)
  }

  *kr { | in1, m(1.0), b0(1.0), bm(0.0) |
      ^this.multiNew('control', in1, m, b0, bm)
  } 

  checkInputs {
    if (rate == 'audio', {
      1.do({|i|
        if (inputs.at(i).rate != 'audio', {
          ^(" input at index " + i + "(" + inputs.at(i) + 
            ") is not audio rate");
        });
      });
    });
    ^this.checkValidInputs
  }

  name { ^"FaustFfcomb" }
}

