#include "process_args.h"

Options options;

int process_args(int argc, char **argv) {
  int c;
  options.inputFileName[0] = '\n';
  options.sharedOption = CachePreferNone;
  while (1) {
    static struct option long_options[] = {
      /* These options set a flag. */
      {"calcChange", no_argument, 0, 'c'},
      /* These options don't set a flag.
         We distinguish them by their indices. */
      {"input",  required_argument, 0, 'i'},
      {"inputB",  required_argument, 0, 'ib'},
      {"rmclOptions",  required_argument, 0, 'r'},
      {"shared",  required_argument, 0, 'e'},
      {"maxIters",  required_argument, 0, 'm'},
      {"stride",  required_argument, 0, 'd'},
      {"stats", no_argument, 0, 's'},
      {"ptile",  required_argument, 0, 'p'},
      {"br",  required_argument, 0, 'x'},
      {"bc",  required_argument, 0, 'y'},
      {"nbins",  required_argument, 0, 'nbins'},
      {"hashing",  required_argument, 0, 'hash'},
      {"refinedESC",  required_argument, 0, 'esc'},
      {"memoryManaged", required_argument, 0, 'mm'},
      {"differentMatrices", required_argument, 0, 'dm'},
      {"Bcolpartition", required_argument, 0, 'bcolpart'},
      {"scatterVector", required_argument, 0, 'sv'},
      {"chunkSizeFactor", required_argument, 0, 'csf'},
      {"chunkSize", required_argument, 0, 'cs'},
      {"twophase", required_argument, 0, 'twophase'},
      {"help",   no_argument, 0, 'h'},
      {0, 0, 0, 0}
    };
    /* getopt_long stores the option index here. */
    int option_index = 0;
    c = getopt_long (argc, argv, "cr:i:m:sx:y:h",
        long_options, &option_index);
    /* Detect the end of the options. */
    if (c == -1)
      break;
    string runString;
    string sharedString;
    switch (c) {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if (long_options[option_index].flag != 0)
          break;
        printf ("option %s", long_options[option_index].name);
        if (optarg)
          printf (" with arg %s", optarg);
        printf ("\n");
        break;

      case 'c':
        options.calcChange = true;
        break;
      case 's':
        options.stats = true;
        break;
      case 'i':
        strcpy(options.inputFileName, optarg);
        break;
      case 'ib':
        strcpy(options.inputFileNameB, optarg);
        break;
      case 'r':
        runString = optarg;
        options.rmclOption = runMap[runString];
        break;
      case 'e':
        sharedString = optarg;
        //cout << sharedString << endl;
        options.sharedOption = sharedMap[sharedString];
        //printf("SharedOption= %s %d\t", sharedOptionsStr[options.sharedOption], options.sharedOption);
        break;
      case 'm':
        options.maxIters = atol(optarg);
        break;
      case 'd':
        options.stride = atol(optarg);
        break;
      case 'p':
        options.ptile = atol(optarg);
        break;
      case 'x':
        options.br= atol(optarg);
        break;
      case 'y':
        options.bc= atol(optarg);
        break;
      case 'nbins':
        options.nbins= atol(optarg);
        break;
      case 'hash':
        options.hashing= atol(optarg);
        break;
      case 'esc':
        options.refinedESC= atol(optarg);
        break;
      case 'mm':
        options.memory_managed= atol(optarg);
        break;
      case 'dm':
        options.diffMatrices= atol(optarg);
        break;
      case 'sv':
        options.scatterVector = atol(optarg);
        break;
      case 'csf':
        options.chunkSizeFactor = atol(optarg);
        break;
      case 'cs':
        options.chunkSize = atol(optarg);
        break;
      case 'bcolpart':
        options.Bcolpartition= atol(optarg);
        break;
      case 'twophase':
        options.twophase= atol(optarg);
        break;
      case 'h':
        break;
      case '?':
        /* getopt_long already printed an error message. */
        break;

      default:
        abort ();
    }
  }

  /* Print any remaining command line arguments (not options). */
  if (optind < argc)
  {
    printf ("non-option ARGV-elements: ");
    while (optind < argc)
      printf ("%s ", argv[optind++]);
    putchar ('\n');
  }
  return 0;
}

void print_args() {
/*
  printf("{\t");
  printf("calcChange= %s\t", options.calcChange ? "true" : "false");
  printf("stats= %s\t", options.stats ? "true" : "false");
  printf("inputFileName= %s\t", options.inputFileName);
  printf("inputFileNameB= %s\t", options.inputFileNameB);
  printf("maxIters= %d\t", options.maxIters);
  printf("stride= %d\t", options.stride);
  printf("ptile= %d\t", options.ptile);
  printf("rmclOption= %s\t", runOptionsStr[options.rmclOption]);
  printf("SharedOption= %s\t", sharedOptionsStr[options.sharedOption]);
  printf("Bcolpartition= %d\t", options.Bcolpartition);
  printf("}\n");*/
  printf("%d,%s,", options.maxIters,options.inputFileName);
}
