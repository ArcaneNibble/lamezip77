#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "zlib.h"

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s c|C|s|d input output\n", argv[0]);
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

    char *outp = malloc(inp_len * 1024);

    z_stream zstr;
    zstr.zalloc = zstr.zfree = zstr.opaque = 0;
    zstr.avail_in = inp_len;
    zstr.next_in = inp;
    zstr.avail_out = inp_len * 1024;
    zstr.next_out = outp;

    if (argv[1][0] == 'c')
    {
        deflateInit2(&zstr, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
        deflate(&zstr, Z_FINISH);
    }
    else if (argv[1][0] == 'C')
    {
        deflateInit2(&zstr, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8, Z_FIXED);
        deflate(&zstr, Z_FINISH);
    }
    else if (argv[1][0] == 's')
    {
        deflateInit2(&zstr, Z_NO_COMPRESSION, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
        deflate(&zstr, Z_FINISH);
    }
    else if (argv[1][0] == 'd')
    {
        inflateInit2(&zstr, -15);
        inflate(&zstr, Z_FINISH);
    }

    FILE *outf = fopen(argv[3], "wb");
    fwrite(outp, 1, inp_len * 1024 - zstr.avail_out, outf);
    fclose(outf);

    return 0;
}
