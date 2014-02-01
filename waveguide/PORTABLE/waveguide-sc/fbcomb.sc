FaustFbackComb : UGen
{
  *ar { | in1, b0(1.0), bm(0.0), del(1.0) |
      ^this.multiNew('audio', in1, b0, bm, del)
  }

  *kr { | in1, b0(1.0), bm(0.0), del(1.0) |
      ^this.multiNew('control', in1, b0, bm, del)
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

  name { ^"FaustFbackComb" }
}

