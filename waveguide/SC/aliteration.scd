(
~dcshift = {|in, amount|
    FreqShift.ar(in, amount);
};

~resonator_analysis = {|bufpath, size|
    PartConv.getBuffer(bufpath, size);
}

)
