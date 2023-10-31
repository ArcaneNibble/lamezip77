#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "fastlz.h"

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s c|C|d input output\n", argv[0]);
        return -1;
    }

    FILE *inf = fopen(argv[2], "rb");
    fseek(inf, 0, SEEK_END);
    size_t inp_len = ftell(inf);
    fseek(inf, 0, SEEK_SET);
    char *inp = malloc(inp_len);
    size_t ret = fread(inp, 1, inp_len, inf);
    assert(ret == inp_len);
    fclose(inf);

    char *outp = malloc(inp_len * 10);

    if (argv[1][0] == 'c')
    {
        ret = fastlz_compress_level(1, inp, inp_len, outp);
    }
    else if (argv[1][0] == 'C')
    {
        ret = fastlz_compress_level(2, inp, inp_len, outp);
    }
    else if (argv[1][0] == 'd')
    {
        ret = fastlz_decompress(inp, inp_len, outp, inp_len * 10);
    }

    FILE *outf = fopen(argv[3], "wb");
    fwrite(outp, 1, ret, outf);
    fclose(outf);

    return 0;
}
