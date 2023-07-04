/* varcoeffED.c 
   version 1 (August 2001), as described in the article
	"A Simple and Efficient Error-Diffusion Algorithm" (SIGGRAPH'01)
   Author: Victor Ostromoukhov
   University of Montreal, http://www.iro.umontreal.ca/~ostrom/

   Usage: varcoeffED
        
   Action:
	- reading PGM input file "input.pgm"
	- making halftone image of size DEFAULT_OUTPUT_DIMS_X by
                                        DEFAULT_OUTPUT_DIMS_Y
	- writing PGM output file "output.pgm"

   Structrure:
    t_image *read_PGM(char *fname)
    void write_PGM(t_image *image, char *fname)
    void allocate_image(t_image *image)
    void shift_carry_buffers()
    void distribute_error(int x, int y, t_carry diff, int dir, int input_level)
    void distribute_error_fs(int x, int y, t_carry diff, int dir)
    void make_output()
    int main(argc, argv)
*******************************************************************/

#include <stdio.h>
#include <math.h>

/*------------------ TYPES ------------------*/
typedef unsigned char t_data;

typedef double t_carry;

typedef struct t_image {
  unsigned int        xdim;                           /* image width */
  unsigned int        ydim;                           /* image height */
  t_data      **data;
} t_image;

typedef struct t_three_coefs {
        int i_r;        /* right */
        int i_dl;       /* down-left */
        int i_d;        /* down */
        int i_sum;      /* sum */
} t_three_coefs;

/*------------------ SOME PARAMETERS, MACROS, CONSTANTS ------------------*/
#ifndef TRUE
#define TRUE  1
#define FALSE  0
#endif

#define TEST_EVEN(x)    (((x) & 1) == 0)

#define DEFAULT_OUTPUT_DIMS_X	256
#define DEFAULT_OUTPUT_DIMS_Y	256

#define TO_RIGHT        1    /* 2 possible directions in boustrophedon mode */
#define TO_LEFT         -1

#define BLACK		0
#define WHITE		255

/*-------------------- GLOBAL DATA -----------------*/
t_image *input_image, *output_image;
char *input_fname = "input.pgm";
char *output_fname = "output.pgm";

t_carry    * carry_line_0 = NULL;       /* carry buffer; current line     */
t_carry    * carry_line_1 = NULL;       /* carry buffer; current line + 1 */

extern void eval_args();
extern void init();
extern void make_output();
extern void fill_image();
extern void allocate_image(t_image *image);

/*-------------------- PROCEDURES -----------------*/
t_image *read_PGM(char *fname)
{
  int byte, nitems;
  int  row, col;
  char s[1024];
  FILE * image_file;
  int ASCII_flag = FALSE;
  t_image *image;
  int dims_x, dims_y;

  sprintf(s,"%s.pgm",fname);
  image_file = fopen(s, "r");
  if (image_file == NULL) {
    sprintf(s,"%s",fname);
    image_file = fopen(s, "r");
    if (image_file == NULL) {
      fprintf(stderr, "Error: Cannot open image file %s\n",s);
      exit(1);
    }
  }
  fflush(stderr);
  fgets(s, 255, image_file);
  if (strncmp(s,"P2",2) == 0) ASCII_flag = TRUE;
  if (strncmp(s,"P2",2) == 0 ||  /* ASCII PGM file */
      strncmp(s,"P5",2) == 0 ) { /* BINARY PGM file */
    fgets(s, 255, image_file);
    while (s[0] == '#') fgets(s, 255, image_file); /* Skip comments */
    sscanf(s,"%d %d", &dims_x, &dims_y);
    fgets(s, 255, image_file); /* Nb of gray levels */
    while (s[0] == '#') fgets(s, 255, image_file); /* Skip comments */
    image = (t_image *)malloc(sizeof(t_image));
    image->xdim = dims_x;
    image->ydim = dims_y;
    allocate_image(image);
    fprintf(stderr, "Reading %s (PGM %dx%d)...", fname,dims_x,dims_y);
    fflush(stderr);
    for (row=0; row<image->ydim; row++) {
      for (col=0; col<image->xdim; col++) {
        if (ASCII_flag) {
          nitems = fscanf(image_file, "%d", &byte);
          if (nitems == 0) {
	    fprintf(stderr, "Error during reading ASCII PGM input file\n");
	    exit(1);
          }
        } else {
          byte = fgetc(image_file);
          if (byte == EOF) {
	    fprintf(stderr, "Error during reading BINARY PGM input file\n");
	    exit(1);
          }
        }
	image->data[row][col] = (unsigned char)byte;
      }
    }
    fclose(image_file);
    fprintf(stderr, "done.\n");
  } else {
    fprintf(stderr,"read_PGM: unknown file type %s\n",s);
    exit(1);
  }
  return image;
} /* read_PGM */

void write_PGM(t_image *image, char *fname)
{
  FILE *out;
  int x, y;

  out = fopen(fname, "w+");
  fprintf(stderr,"Writing %s (PGM %dx%d)...",fname,image->xdim,image->ydim);
  fflush(stderr);
  fprintf(out, "P5\n%d %d\n%d\n", image->xdim, image->ydim, 255);
  fflush(out);
  for(y = 0; y < image->ydim; y++) {
    for(x = 0; x < image->xdim; x++) {
      fprintf(out, "%c", image->data[y][x]);
    }
  } 
  fclose(out);
  fprintf(stderr,"done.\n");fflush(stderr);
} /* write_PGM */

/*------------------------ memory allocation ------------------------*/
void allocate_image(t_image *image)
{
  int row;
  int line_length = image->xdim*sizeof(t_data);
  image->data = (t_data **)malloc(image->ydim*sizeof(t_data *));
  for (row = 0; row < image->ydim; row++) {
    image->data[row] = (t_data *)malloc(line_length);
  }
} /* allocate_image */

void shift_carry_buffers()
{
  t_carry *tmp;
  tmp=carry_line_0;
  carry_line_0 = carry_line_1;
  carry_line_1 = tmp;
  memset (carry_line_1, 0, DEFAULT_OUTPUT_DIMS_X*sizeof(t_carry));
}

/*---------------- varcoeff -----------------*/
t_three_coefs var_coefs_tab[256] = {
    13,     0,     5,    18,     /*    0 */
    13,     0,     5,    18,     /*    1 */
    21,     0,    10,    31,     /*    2 */
     7,     0,     4,    11,     /*    3 */
     8,     0,     5,    13,     /*    4 */
    47,     3,    28,    78,     /*    5 */
    23,     3,    13,    39,     /*    6 */
    15,     3,     8,    26,     /*    7 */
    22,     6,    11,    39,     /*    8 */
    43,    15,    20,    78,     /*    9 */
     7,     3,     3,    13,     /*   10 */
   501,   224,   211,   936,     /*   11 */
   249,   116,   103,   468,     /*   12 */
   165,    80,    67,   312,     /*   13 */
   123,    62,    49,   234,     /*   14 */
   489,   256,   191,   936,     /*   15 */
    81,    44,    31,   156,     /*   16 */
   483,   272,   181,   936,     /*   17 */
    60,    35,    22,   117,     /*   18 */
    53,    32,    19,   104,     /*   19 */
   237,   148,    83,   468,     /*   20 */
   471,   304,   161,   936,     /*   21 */
     3,     2,     1,     6,     /*   22 */
   459,   304,   161,   924,     /*   23 */
    38,    25,    14,    77,     /*   24 */
   453,   296,   175,   924,     /*   25 */
   225,   146,    91,   462,     /*   26 */
   149,    96,    63,   308,     /*   27 */
   111,    71,    49,   231,     /*   28 */
    63,    40,    29,   132,     /*   29 */
    73,    46,    35,   154,     /*   30 */
   435,   272,   217,   924,     /*   31 */
   108,    67,    56,   231,     /*   32 */
    13,     8,     7,    28,     /*   33 */
   213,   130,   119,   462,     /*   34 */
   423,   256,   245,   924,     /*   35 */
     5,     3,     3,    11,     /*   36 */
   281,   173,   162,   616,     /*   37 */
   141,    89,    78,   308,     /*   38 */
   283,   183,   150,   616,     /*   39 */
    71,    47,    36,   154,     /*   40 */
   285,   193,   138,   616,     /*   41 */
    13,     9,     6,    28,     /*   42 */
    41,    29,    18,    88,     /*   43 */
    36,    26,    15,    77,     /*   44 */
   289,   213,   114,   616,     /*   45 */
   145,   109,    54,   308,     /*   46 */
   291,   223,   102,   616,     /*   47 */
    73,    57,    24,   154,     /*   48 */
   293,   233,    90,   616,     /*   49 */
    21,    17,     6,    44,     /*   50 */
   295,   243,    78,   616,     /*   51 */
    37,    31,     9,    77,     /*   52 */
    27,    23,     6,    56,     /*   53 */
   149,   129,    30,   308,     /*   54 */
   299,   263,    54,   616,     /*   55 */
    75,    67,    12,   154,     /*   56 */
    43,    39,     6,    88,     /*   57 */
   151,   139,    18,   308,     /*   58 */
   303,   283,    30,   616,     /*   59 */
    38,    36,     3,    77,     /*   60 */
   305,   293,    18,   616,     /*   61 */
   153,   149,     6,   308,     /*   62 */
   307,   303,     6,   616,     /*   63 */
     1,     1,     0,     2,     /*   64 */
   101,   105,     2,   208,     /*   65 */
    49,    53,     2,   104,     /*   66 */
    95,   107,     6,   208,     /*   67 */
    23,    27,     2,    52,     /*   68 */
    89,   109,    10,   208,     /*   69 */
    43,    55,     6,   104,     /*   70 */
    83,   111,    14,   208,     /*   71 */
     5,     7,     1,    13,     /*   72 */
   172,   181,    37,   390,     /*   73 */
    97,    76,    22,   195,     /*   74 */
    72,    41,    17,   130,     /*   75 */
   119,    47,    29,   195,     /*   76 */
     4,     1,     1,     6,     /*   77 */
     4,     1,     1,     6,     /*   78 */
     4,     1,     1,     6,     /*   79 */
     4,     1,     1,     6,     /*   80 */
     4,     1,     1,     6,     /*   81 */
     4,     1,     1,     6,     /*   82 */
     4,     1,     1,     6,     /*   83 */
     4,     1,     1,     6,     /*   84 */
     4,     1,     1,     6,     /*   85 */
    65,    18,    17,   100,     /*   86 */
    95,    29,    26,   150,     /*   87 */
   185,    62,    53,   300,     /*   88 */
    30,    11,     9,    50,     /*   89 */
    35,    14,    11,    60,     /*   90 */
    85,    37,    28,   150,     /*   91 */
    55,    26,    19,   100,     /*   92 */
    80,    41,    29,   150,     /*   93 */
   155,    86,    59,   300,     /*   94 */
     5,     3,     2,    10,     /*   95 */
     5,     3,     2,    10,     /*   96 */
     5,     3,     2,    10,     /*   97 */
     5,     3,     2,    10,     /*   98 */
     5,     3,     2,    10,     /*   99 */
     5,     3,     2,    10,     /*  100 */
     5,     3,     2,    10,     /*  101 */
     5,     3,     2,    10,     /*  102 */
     5,     3,     2,    10,     /*  103 */
     5,     3,     2,    10,     /*  104 */
     5,     3,     2,    10,     /*  105 */
     5,     3,     2,    10,     /*  106 */
     5,     3,     2,    10,     /*  107 */
   305,   176,   119,   600,     /*  108 */
   155,    86,    59,   300,     /*  109 */
   105,    56,    39,   200,     /*  110 */
    80,    41,    29,   150,     /*  111 */
    65,    32,    23,   120,     /*  112 */
    55,    26,    19,   100,     /*  113 */
   335,   152,   113,   600,     /*  114 */
    85,    37,    28,   150,     /*  115 */
   115,    48,    37,   200,     /*  116 */
    35,    14,    11,    60,     /*  117 */
   355,   136,   109,   600,     /*  118 */
    30,    11,     9,    50,     /*  119 */
   365,   128,   107,   600,     /*  120 */
   185,    62,    53,   300,     /*  121 */
    25,     8,     7,    40,     /*  122 */
    95,    29,    26,   150,     /*  123 */
   385,   112,   103,   600,     /*  124 */
    65,    18,    17,   100,     /*  125 */
   395,   104,   101,   600,     /*  126 */
     4,     1,     1,     6,     /*  127 */
     4,     1,     1,     6,     /*  128 */
   395,   104,   101,   600,     /*  129 */
    65,    18,    17,   100,     /*  130 */
   385,   112,   103,   600,     /*  131 */
    95,    29,    26,   150,     /*  132 */
    25,     8,     7,    40,     /*  133 */
   185,    62,    53,   300,     /*  134 */
   365,   128,   107,   600,     /*  135 */
    30,    11,     9,    50,     /*  136 */
   355,   136,   109,   600,     /*  137 */
    35,    14,    11,    60,     /*  138 */
   115,    48,    37,   200,     /*  139 */
    85,    37,    28,   150,     /*  140 */
   335,   152,   113,   600,     /*  141 */
    55,    26,    19,   100,     /*  142 */
    65,    32,    23,   120,     /*  143 */
    80,    41,    29,   150,     /*  144 */
   105,    56,    39,   200,     /*  145 */
   155,    86,    59,   300,     /*  146 */
   305,   176,   119,   600,     /*  147 */
     5,     3,     2,    10,     /*  148 */
     5,     3,     2,    10,     /*  149 */
     5,     3,     2,    10,     /*  150 */
     5,     3,     2,    10,     /*  151 */
     5,     3,     2,    10,     /*  152 */
     5,     3,     2,    10,     /*  153 */
     5,     3,     2,    10,     /*  154 */
     5,     3,     2,    10,     /*  155 */
     5,     3,     2,    10,     /*  156 */
     5,     3,     2,    10,     /*  157 */
     5,     3,     2,    10,     /*  158 */
     5,     3,     2,    10,     /*  159 */
     5,     3,     2,    10,     /*  160 */
   155,    86,    59,   300,     /*  161 */
    80,    41,    29,   150,     /*  162 */
    55,    26,    19,   100,     /*  163 */
    85,    37,    28,   150,     /*  164 */
    35,    14,    11,    60,     /*  165 */
    30,    11,     9,    50,     /*  166 */
   185,    62,    53,   300,     /*  167 */
    95,    29,    26,   150,     /*  168 */
    65,    18,    17,   100,     /*  169 */
     4,     1,     1,     6,     /*  170 */
     4,     1,     1,     6,     /*  171 */
     4,     1,     1,     6,     /*  172 */
     4,     1,     1,     6,     /*  173 */
     4,     1,     1,     6,     /*  174 */
     4,     1,     1,     6,     /*  175 */
     4,     1,     1,     6,     /*  176 */
     4,     1,     1,     6,     /*  177 */
     4,     1,     1,     6,     /*  178 */
   119,    47,    29,   195,     /*  179 */
    72,    41,    17,   130,     /*  180 */
    97,    76,    22,   195,     /*  181 */
   172,   181,    37,   390,     /*  182 */
     5,     7,     1,    13,     /*  183 */
    83,   111,    14,   208,     /*  184 */
    43,    55,     6,   104,     /*  185 */
    89,   109,    10,   208,     /*  186 */
    23,    27,     2,    52,     /*  187 */
    95,   107,     6,   208,     /*  188 */
    49,    53,     2,   104,     /*  189 */
   101,   105,     2,   208,     /*  190 */
     1,     1,     0,     2,     /*  191 */
   307,   303,     6,   616,     /*  192 */
   153,   149,     6,   308,     /*  193 */
   305,   293,    18,   616,     /*  194 */
    38,    36,     3,    77,     /*  195 */
   303,   283,    30,   616,     /*  196 */
   151,   139,    18,   308,     /*  197 */
    43,    39,     6,    88,     /*  198 */
    75,    67,    12,   154,     /*  199 */
   299,   263,    54,   616,     /*  200 */
   149,   129,    30,   308,     /*  201 */
    27,    23,     6,    56,     /*  202 */
    37,    31,     9,    77,     /*  203 */
   295,   243,    78,   616,     /*  204 */
    21,    17,     6,    44,     /*  205 */
   293,   233,    90,   616,     /*  206 */
    73,    57,    24,   154,     /*  207 */
   291,   223,   102,   616,     /*  208 */
   145,   109,    54,   308,     /*  209 */
   289,   213,   114,   616,     /*  210 */
    36,    26,    15,    77,     /*  211 */
    41,    29,    18,    88,     /*  212 */
    13,     9,     6,    28,     /*  213 */
   285,   193,   138,   616,     /*  214 */
    71,    47,    36,   154,     /*  215 */
   283,   183,   150,   616,     /*  216 */
   141,    89,    78,   308,     /*  217 */
   281,   173,   162,   616,     /*  218 */
     5,     3,     3,    11,     /*  219 */
   423,   256,   245,   924,     /*  220 */
   213,   130,   119,   462,     /*  221 */
    13,     8,     7,    28,     /*  222 */
   108,    67,    56,   231,     /*  223 */
   435,   272,   217,   924,     /*  224 */
    73,    46,    35,   154,     /*  225 */
    63,    40,    29,   132,     /*  226 */
   111,    71,    49,   231,     /*  227 */
   149,    96,    63,   308,     /*  228 */
   225,   146,    91,   462,     /*  229 */
   453,   296,   175,   924,     /*  230 */
    38,    25,    14,    77,     /*  231 */
   459,   304,   161,   924,     /*  232 */
     3,     2,     1,     6,     /*  233 */
   471,   304,   161,   936,     /*  234 */
   237,   148,    83,   468,     /*  235 */
    53,    32,    19,   104,     /*  236 */
    60,    35,    22,   117,     /*  237 */
   483,   272,   181,   936,     /*  238 */
    81,    44,    31,   156,     /*  239 */
   489,   256,   191,   936,     /*  240 */
   123,    62,    49,   234,     /*  241 */
   165,    80,    67,   312,     /*  242 */
   249,   116,   103,   468,     /*  243 */
   501,   224,   211,   936,     /*  244 */
     7,     3,     3,    13,     /*  245 */
    43,    15,    20,    78,     /*  246 */
    22,     6,    11,    39,     /*  247 */
    15,     3,     8,    26,     /*  248 */
    23,     3,    13,    39,     /*  249 */
    47,     3,    28,    78,     /*  250 */
     8,     0,     5,    13,     /*  251 */
     7,     0,     4,    11,     /*  252 */
    21,     0,    10,    31,     /*  253 */
    13,     0,     5,    18,     /*  254 */
    13,     0,     5,    18};    /*  255 */

#define SET_CARRY_0(x,val) { carry_line_0[x] += (t_carry) val; }
#define SET_CARRY_1(x,val) { carry_line_1[x] += (t_carry) val; }

void distribute_error(int x, int y, t_carry diff, int dir, int input_level)
{
  t_carry term_r, term_dl, term_d;
  t_three_coefs coefs = var_coefs_tab[input_level];

  term_r = (t_carry)coefs.i_r*diff/(t_carry)coefs.i_sum;
  term_dl = (t_carry)coefs.i_dl*diff/(t_carry)coefs.i_sum;
  term_d = diff - (term_r+term_dl);

  SET_CARRY_0(x+dir, term_r);
  SET_CARRY_1(x-dir, term_dl);
  SET_CARRY_1(x,     term_d);
} /* distribute_error */

/*---------------- FLOYD-STEINBERG -----------------*/
#define SET_CARRY_7351_0(x, diff) carry_line_0[x] += (t_carry) (diff/16.)
#define SET_CARRY_7351_1(x, diff) carry_line_1[x] += (t_carry) (diff/16.)

void distribute_error_fs(int x, int y, t_carry diff, int dir)
{
  t_carry   diff_2 = diff + diff ;
  t_carry   diff_3 = diff + diff_2 ;
  t_carry   diff_5 = diff_3 + diff_2 ;
  t_carry   diff_7 = diff_5 + diff_2 ;
  SET_CARRY_7351_0(x+dir, diff_7);
  SET_CARRY_7351_1(x-dir, diff_3);
  SET_CARRY_7351_1(x,            diff_5);
  SET_CARRY_7351_1(x+dir, diff);
} /* distribute_error_fs */
/*---------------- END FLOYD-STEINBERG -----------------*/

void make_output()
{
  int x, y, xsrc, ysrc, *x_tab, *y_tab;
  int xstart, xstop, xstep, ystart, ystop, ystep, dir;
  double x_scale_factor, y_scale_factor;
  int xdim_in = input_image->xdim, ydim_in = input_image->ydim;
  int xdim_out = DEFAULT_OUTPUT_DIMS_X, ydim_out = DEFAULT_OUTPUT_DIMS_Y;
  int input, intensity;
  t_carry threshold = 127.5, corrected_level, diff;

  /* allocate carry_line_0 and carry_line_1 */
  carry_line_0 = 1+(t_carry *)calloc(xdim_out+2,sizeof(t_carry));
  carry_line_1 = 1+(t_carry *)calloc(xdim_out+2,sizeof(t_carry));

  /* allocate and init x_tab and y_tab */
  x_tab = (int *)calloc(xdim_out, sizeof(int));
  y_tab = (int *)calloc(ydim_out, sizeof(int));
  x_scale_factor = (double)xdim_out / (double)(xdim_in);
  y_scale_factor = (double)ydim_out / (double)(ydim_in);
  for (x = 0; x < xdim_out; x++)
    x_tab[x] = floor(x / x_scale_factor);
  for (y = 0; y < ydim_out; y++) {
    y_tab[y] = floor(y / y_scale_factor);
  }

  fprintf(stderr, "Making output image...\n");

  ystart = 0; ystop = ydim_out;
  for (y = ystart; y < ystop; y++) {
    if (TEST_EVEN(y)) {
      dir = TO_RIGHT;
      xstart = 0; xstop = xdim_out; xstep = 1;
    } else { /* even lines */
      dir = TO_LEFT;
      xstart = xdim_out-1; xstop = -1; xstep = -1;
    } /* if (TEST_EVEN(y)) */

    ysrc = y_tab[y];
    for (x = xstart; x != xstop; x += xstep) {
      xsrc = x_tab[x];
      input = input_image->data[ysrc][xsrc];
      corrected_level = input + carry_line_0[x];
      if (corrected_level <= threshold)
	intensity = BLACK; /* put black */
      else
	intensity = WHITE; /* put white */
      diff = corrected_level - intensity;
      distribute_error(x, y, diff, dir, input);

      if (input == BLACK || intensity == BLACK) 
	output_image->data[y][x] = BLACK;
      else
	output_image->data[y][x] = WHITE;
    } /* x-cycle */
    shift_carry_buffers();
  } /* y-cycle */

}  /* make_output */


int main(int argc, char **argv)
{
  if (argc < 3) {
    printf("Too few arguments. [input file] [output file] [size].");
    return 1;
  }

  input_image = read_PGM(argv[1]);
  output_image = (t_image *)malloc(sizeof(t_image));
  output_image->xdim = DEFAULT_OUTPUT_DIMS_X;
  output_image->ydim = DEFAULT_OUTPUT_DIMS_Y;
  allocate_image(output_image);
  make_output();
  write_PGM(output_image, argv[2]);
} /* main */
