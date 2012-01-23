/*-----------------------------------------------------------------------*/
/* Program: Stream                                                       */
/* Revision: $Id: stream.c,v 5.9 2009/04/11 16:35:00 mccalpin Exp $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2005: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*         "tuned STREAM benchmark results"                              */
/*         "based on a variant of the STREAM benchmark code"             */
/*         Other comparable, clear and reasonable labelling is           */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/

/** @file stream.c
 * This example demonstrates loading, running and scripting a very simple
 * NaCl module.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/nacl_syscalls.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdarg.h>

#include "ppapi/c/pp_errors.h"
#include "ppapi/c/pp_module.h"
#include "ppapi/c/pp_var.h"
#include "ppapi/c/ppb.h"
#include "ppapi/c/ppb_instance.h"
#include "ppapi/c/ppb_messaging.h"
#include "ppapi/c/ppb_var.h"
#include "ppapi/c/ppp.h"
#include "ppapi/c/ppp_instance.h"
#include "ppapi/c/ppp_messaging.h"

/* INSTRUCTIONS:
 *
 *	1) Stream requires a good bit of memory to run.  Adjust the
 *          value of 'N' (below) to give a 'timing calibration' of 
 *          at least 20 clock-ticks.  This will provide rate estimates
 *          that should be good to about 5% precision.
 */
#define DEBUG

#ifndef N
#   define N	40000000
#endif
#ifndef NTIMES
#   define NTIMES	10
#endif
#ifndef OFFSET
#   define OFFSET	0
#endif

/*
 *	3) Compile the code with full optimization.  Many compilers
 *	   generate unreasonably bad code before the optimizer tightens
 *	   things up.  If the results are unreasonably good, on the
 *	   other hand, the optimizer might be too smart for me!
 *
 *         Try compiling with:
 *               cc -O stream_omp.c -o stream_omp
 *
 *         This is known to work on Cray, SGI, IBM, and Sun machines.
 *
 *
 *	4) Mail the results to mccalpin@cs.virginia.edu
 *	   Be sure to include:
 *		a) computer hardware model number and software revision
 *		b) the compiler flags
 *		c) all of the output from the test case.
 * Thanks!
 *
 */

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

static double	a[N+OFFSET],
		b[N+OFFSET],
		c[N+OFFSET];

static double	avgtime[4] = {0}, maxtime[4] = {0},
		mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[4] = {"Copy:      ", "Scale:     ",
    "Add:       ", "Triad:     "};

static double	bytes[4] = {
    2 * sizeof(double) * N,
    2 * sizeof(double) * N,
    3 * sizeof(double) * N,
    3 * sizeof(double) * N
    };

extern double mysecond();
extern void checkSTREAMresults();
#ifdef TUNED
extern void tuned_STREAM_Copy();
extern void tuned_STREAM_Scale(double scalar);
extern void tuned_STREAM_Add();
extern void tuned_STREAM_Triad(double scalar);
#endif
#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

static PP_Module module_id = 0;
static struct PPB_Messaging* messaging_interface = NULL;
static struct PPB_Var* var_interface = NULL;

static const char* const kStreamMethodId = "stream";
static const char kMessageArgumentSeparator = ':';
static const char kNullTerminator = '\0';

void get_str(long, int, char *);
int get_separator_at(char *);

char oStr[1000] = "";
char tStr[100] = "";
char sep = ':';


/**
 * Returns a mutable C string contained in the @a var or NULL if @a var is not
 * string.  This makes a copy of the string in the @a var and adds a NULL
 * terminator.  Note that VarToUtf8() does not guarantee the NULL terminator on
 * the returned string.  See the comments for VarToUtf8() in ppapi/c/ppb_var.h
 * for more info.  The caller is responsible for freeing the returned memory.
 * @param[in] var PP_Var containing string.
 * @return a mutable C string representation of @a var.
 * @note The caller is responsible for freeing the returned string.
 */
/* TODO(sdk_user): 2. Uncomment this when you need it.  It is commented out so
 * that the compiler doesn't complain about unused functions.
 */

static char* AllocateCStrFromVar(struct PP_Var var) {
  uint32_t len = 0;
  if (var_interface != NULL) {
    const char* var_c_str = var_interface->VarToUtf8(var, &len);
    if (len > 0) {
      char* c_str = (char*)malloc(len + 1);
      memcpy(c_str, var_c_str, len);
      c_str[len] = '\0';
      return c_str;
    }
  }
  return NULL;
}


/**
 * Creates a new string PP_Var from C string. The resulting object will be a
 * refcounted string object. It will be AddRef()ed for the caller. When the
 * caller is done with it, it should be Release()d.
 * @param[in] str C string to be converted to PP_Var
 * @return PP_Var containing string.
 */
/* TODO(sdk_user): 3. Uncomment this when you need it.  It is commented out so
 * that the compiler doesn't complain about unused functions.
 */

static struct PP_Var AllocateVarFromCStr(const char* str) {
  if (var_interface != NULL)
    return var_interface->VarFromUtf8(module_id, str, strlen(str));
  return PP_MakeUndefined();
}


/**
 * Called when the NaCl module is instantiated on the web page. The identifier
 * of the new instance will be passed in as the first argument (this value is
 * generated by the browser and is an opaque handle).  This is called for each
 * instantiation of the NaCl module, which is each time the <embed> tag for
 * this module is encountered.
 *
 * If this function reports a failure (by returning @a PP_FALSE), the NaCl
 * module will be deleted and DidDestroy will be called.
 * @param[in] instance The identifier of the new instance representing this
 *     NaCl module.
 * @param[in] argc The number of arguments contained in @a argn and @a argv.
 * @param[in] argn An array of argument names.  These argument names are
 *     supplied in the <embed> tag, for example:
 *       <embed id="nacl_module" dimensions="2">
 *     will produce two arguments, one named "id" and one named "dimensions".
 * @param[in] argv An array of argument values.  These are the values of the
 *     arguments listed in the <embed> tag.  In the above example, there will
 *     be two elements in this array, "nacl_module" and "2".  The indices of
 *     these values match the indices of the corresponding names in @a argn.
 * @return @a PP_TRUE on success.
 */
static PP_Bool Instance_DidCreate(PP_Instance instance,
                                  uint32_t argc,
                                  const char* argn[],
                                  const char* argv[]) {
  return PP_TRUE;
}

/**
 * Called when the NaCl module is destroyed. This will always be called,
 * even if DidCreate returned failure. This routine should deallocate any data
 * associated with the instance.
 * @param[in] instance The identifier of the instance representing this NaCl
 *     module.
 */
static void Instance_DidDestroy(PP_Instance instance) {
}

/**
 * Called when the position, the size, or the clip rect of the element in the
 * browser that corresponds to this NaCl module has changed.
 * @param[in] instance The identifier of the instance representing this NaCl
 *     module.
 * @param[in] position The location on the page of this NaCl module. This is
 *     relative to the top left corner of the viewport, which changes as the
 *     page is scrolled.
 * @param[in] clip The visible region of the NaCl module. This is relative to
 *     the top left of the plugin's coordinate system (not the page).  If the
 *     plugin is invisible, @a clip will be (0, 0, 0, 0).
 */
static void Instance_DidChangeView(PP_Instance instance,
                                   const struct PP_Rect* position,
                                   const struct PP_Rect* clip) {
}

/**
 * Notification that the given NaCl module has gained or lost focus.
 * Having focus means that keyboard events will be sent to the NaCl module
 * represented by @a instance. A NaCl module's default condition is that it
 * will not have focus.
 *
 * Note: clicks on NaCl modules will give focus only if you handle the
 * click event. You signal if you handled it by returning @a true from
 * HandleInputEvent. Otherwise the browser will bubble the event and give
 * focus to the element on the page that actually did end up consuming it.
 * If you're not getting focus, check to make sure you're returning true from
 * the mouse click in HandleInputEvent.
 * @param[in] instance The identifier of the instance representing this NaCl
 *     module.
 * @param[in] has_focus Indicates whether this NaCl module gained or lost
 *     event focus.
 */
static void Instance_DidChangeFocus(PP_Instance instance,
                                    PP_Bool has_focus) {
}

/**
 * Handler that gets called after a full-frame module is instantiated based on
 * registered MIME types.  This function is not called on NaCl modules.  This
 * function is essentially a place-holder for the required function pointer in
 * the PPP_Instance structure.
 * @param[in] instance The identifier of the instance representing this NaCl
 *     module.
 * @param[in] url_loader A PP_Resource an open PPB_URLLoader instance.
 * @return PP_FALSE.
 */
static PP_Bool Instance_HandleDocumentLoad(PP_Instance instance,
                                           PP_Resource url_loader) {
  /* NaCl modules do not need to handle the document load function. */
  return PP_FALSE;
}


/**
 * Handler for messages coming in from the browser via postMessage.  The
 * @a var_message can contain anything: a JSON string; a string that encodes
 * method names and arguments; etc.  For example, you could use JSON.stringify
 * in the browser to create a message that contains a method name and some
 * parameters, something like this:
 *   var json_message = JSON.stringify({ "myMethod" : "3.14159" });
 *   nacl_module.postMessage(json_message);
 * On receipt of this message in @a var_message, you could parse the JSON to
 * retrieve the method name, match it to a function call, and then call it with
 * the parameter.
 * @param[in] instance The instance ID.
 * @param[in] message The contents, copied by value, of the message sent from
 *     browser via postMessage.
 */
void Messaging_HandleMessage(PP_Instance instance, struct PP_Var var_message) {
  /* TODO(sdk_user): 1. Make this function handle the incoming message. */
  if (var_message.type != PP_VARTYPE_STRING) {
    /* Only handle string messages */
    return;
  }
  char* message = AllocateCStrFromVar(var_message);
  if (message == NULL)
    return;
  struct PP_Var var_result = PP_MakeUndefined();
  if (strncmp(message, kStreamMethodId, strlen(kStreamMethodId)) == 0) {
    char* string_arg = strchr(message, kMessageArgumentSeparator);
    if (string_arg != NULL) {
      string_arg += 1;  /* Advance past the ':' separator. */
      nacl_main();
      var_result = AllocateVarFromCStr(oStr);
    }
  } 
  
  free(message);

  /* Echo the return result back to browser.  Note that HandleMessage is always
   * called on the main thread, so it's OK to post the message back to the
   * browser directly from here.  This return post is asynchronous.
   */
  messaging_interface->PostMessage(instance, var_result);
  /* If the message was created using VarFromUtf8() it needs to be released.
   * See the comments about VarFromUtf8() in ppapi/c/ppb_var.h for more
   * information.
   */
  if (var_result.type == PP_VARTYPE_STRING) {
    var_interface->Release(var_result);
  }
}

/**
 * Entry points for the module.
 * Initialize instance interface and scriptable object class.
 * @param[in] a_module_id Module ID
 * @param[in] get_browser_interface Pointer to PPB_GetInterface
 * @return PP_OK on success, any other value on failure.
 */
PP_EXPORT int32_t PPP_InitializeModule(PP_Module a_module_id,
                                       PPB_GetInterface get_browser_interface) {
  module_id = a_module_id;
  var_interface = (struct PPB_Var*)(get_browser_interface(PPB_VAR_INTERFACE));
  messaging_interface =
      (struct PPB_Messaging*)(get_browser_interface(PPB_MESSAGING_INTERFACE));
  return PP_OK;
}

/**
 * Returns an interface pointer for the interface of the given name, or NULL
 * if the interface is not supported.
 * @param[in] interface_name name of the interface
 * @return pointer to the interface
 */
PP_EXPORT const void* PPP_GetInterface(const char* interface_name) {
  if (strcmp(interface_name, PPP_INSTANCE_INTERFACE) == 0) {
    static struct PPP_Instance instance_interface = {
      &Instance_DidCreate,
      &Instance_DidDestroy,
      &Instance_DidChangeView,
      &Instance_DidChangeFocus,
      &Instance_HandleDocumentLoad
    };
    return &instance_interface;
  } else if (strcmp(interface_name, PPP_MESSAGING_INTERFACE) == 0) {
    static struct PPP_Messaging messaging_interface = {
      &Messaging_HandleMessage
    };
    return &messaging_interface;
  }
  return NULL;
}

/**
 * Called before the plugin module is unloaded.
 */
PP_EXPORT void PPP_ShutdownModule() {
}

void nacl_printf(const char *s,  ...) {
#ifdef DEBUG
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    va_end(args);
#endif

#if 0
    tmpStr[0] = sep;
    strcat(tmpStr, s);
    strcat(oStr, tmpStr);
    
    va_list args;
    fprintf( stderr, "Error: " );
    va_start( args, s );
    vfprintf( stderr, a, args );
    va_end( args );
    fprintf( stderr, "\n" );
#endif

}

int nacl_main() {
    int			quantum, checktick();
    int			BytesPerWord;
    register int	j, k;
    double		scalar, t, times[4][NTIMES];
    

    /* --- SETUP --- determine precision and check timing --- */
    nacl_printf("-------------------------------------------------------------\n");
    nacl_printf("STREAM version $Revision: 5.9 $\n");
    nacl_printf("-------------------------------------------------------------\n");
    BytesPerWord = sizeof(double);
    nacl_printf("This system uses %d bytes per DOUBLE PRECISION word.\n", BytesPerWord);

    nacl_printf("------------------------------------------------------------\n");
#ifdef NO_LONG_LONG
    nacl_printf(tStr, "Array size = %d, Offset = %d\n", N, OFFSET);
#else
    nacl_printf("Array size = %llu, Offset = %d\n", (unsigned long long)N, OFFSET);
#endif

    nacl_printf("Total memory required = %.1f MB.\n",
		(3.0 * BytesPerWord) * ( (double) N / 1048576.0));
    nacl_printf("Each test is run %d times, but only\n", NTIMES);
    nacl_printf("the *best* time for each is used.\n");
	
#ifdef _OPENMP
    strcat(oStr, "-------------------------------------------------------------:");
#pragma omp parallel 
    {
#pragma omp master
	{
	    k = omp_get_num_threads();
	    nacl_printf ("Number of Threads requested = %i\n", k);
    }
    }
#endif

    nacl_printf("------------------------------------------------------------\n");
#pragma omp parallel
    {
    nacl_printf ("Printing one line per active thread....\n");
    }

    /* Get initial value for system clock. */
#pragma omp parallel for
    for (j=0; j<N; j++) {
	a[j] = 1.0;
	b[j] = 2.0;
	c[j] = 0.0;
	}

    nacl_printf("------------------------------------------------------------\n");
    
    if  ( (quantum = checktick()) >= 1) {
		nacl_printf("Your clock granularity/precision appears to be "
	    	"%d microseconds.\n", quantum);
    }
    
    else {
		nacl_printf("Your clock granularity appears to be "
		    "less than one microsecond.\n");
		quantum = 1;
    }

    t = mysecond();
#pragma omp parallel for
    for (j = 0; j < N; j++)
	a[j] = 2.0E0 * a[j];
    t = 1.0E6 * (mysecond() - t);

    nacl_printf("Each test below will take on the order"
	" of %d microseconds.\n", (int)t);
    nacl_printf("   (= %d clock ticks)\n", (int)(t/quantum));
    nacl_printf("Increase the size of the arrays if this shows that\n");
    nacl_printf("you are not getting at least 20 clock ticks per test.\n");
	
    nacl_printf("------------------------------------------------------------\n");

    nacl_printf("WARNING -- The above is only a rough guideline.\n");
    nacl_printf("For best results, please be sure you know the\n");
    nacl_printf("precision of your system timer.\n");
    nacl_printf("------------------------------------------------------------\n");
    
    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

    scalar = 3.0;
    for (k=0; k<NTIMES; k++)
	{
	times[0][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Copy();
#else
#pragma omp parallel for
	for (j=0; j<N; j++)
	    c[j] = a[j];
#endif
	times[0][k] = mysecond() - times[0][k];
	
	times[1][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Scale(scalar);
#else
#pragma omp parallel for
	for (j=0; j<N; j++)
	    b[j] = scalar*c[j];
#endif
	times[1][k] = mysecond() - times[1][k];
	
	times[2][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Add();
#else
#pragma omp parallel for
	for (j=0; j<N; j++)
	    c[j] = a[j]+b[j];
#endif
	times[2][k] = mysecond() - times[2][k];
	
	times[3][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#pragma omp parallel for
	for (j=0; j<N; j++)
	    a[j] = b[j]+scalar*c[j];
#endif
	times[3][k] = mysecond() - times[3][k];
	}

    /*	--- SUMMARY --- */

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
	for (j=0; j<4; j++)
	    {
	    avgtime[j] = avgtime[j] + times[j][k];
	    mintime[j] = MIN(mintime[j], times[j][k]);
	    maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}
    
    nacl_printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
    for (j=0; j<4; j++) {
	avgtime[j] = avgtime[j]/(double)(NTIMES-1);

	nacl_printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
	       1.0E-06 * bytes[j]/mintime[j],
	       avgtime[j],
	       mintime[j],
	       maxtime[j]);
    }
    nacl_printf("------------------------------------------------------------\n");

    /* --- Check Results --- */
    checkSTREAMresults();
    nacl_printf("------------------------------------------------------------\n");

    return 0;
}

# define	M	20

int
checktick()
    {
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++) {
	t1 = mysecond();
	while( ((t2=mysecond()) - t1) < 1.0E-6 )
	    ;
	timesfound[i] = t1 = t2;
	}

/*
 * Determine the minimum difference between these M values.
 * This result will be our estimate (in microseconds) for the
 * clock granularity.
 */

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
	Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
	minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
    }



/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void checkSTREAMresults ()
{
	double aj,bj,cj,scalar;
	double asum,bsum,csum;
	double epsilon;
	int	j,k;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;
    /* now execute timing loop */
	scalar = 3.0;
	for (k=0; k<NTIMES; k++)
        {
            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;
        }
	aj = aj * (double) (N);
	bj = bj * (double) (N);
	cj = cj * (double) (N);

	asum = 0.0;
	bsum = 0.0;
	csum = 0.0;
	for (j=0; j<N; j++) {
		asum += a[j];
		bsum += b[j];
		csum += c[j];
	}
#ifdef VERBOSE
	nacl_printf("Results Comparison: \n");
	nacl_printf("        Expected  : %f %f %f \n",aj,bj,cj);
	nacl_printf("        Observed  : %f %f %f \n",asum,bsum,csum);
#endif

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
	epsilon = 1.e-8;

	if (abs(aj-asum)/asum > epsilon) {
		nacl_printf ("Failed Validation on array a[]\n");
		nacl_printf ("        Expected  : %f \n",aj);
		nacl_printf ("        Observed  : %f \n",asum);
	}
	else if (abs(bj-bsum)/bsum > epsilon) {
		nacl_printf("Failed Validation on array b[]\n");
		nacl_printf("        Expected  : %f \n",bj);
		nacl_printf("        Observed  : %f \n",bsum);
	}
	else if (abs(cj-csum)/csum > epsilon) {
		nacl_printf ("Failed Validation on array c[]\n");
		nacl_printf ("        Expected  : %f \n",cj);
		nacl_printf ("        Observed  : %f \n",csum);
	}
	else {
		nacl_printf ("Solution Validates\n");
	}
}

void tuned_STREAM_Copy()
{
	int j;
#pragma omp parallel for
        for (j=0; j<N; j++)
            c[j] = a[j];
}

void tuned_STREAM_Scale(double scalar)
{
	int j;
#pragma omp parallel for
	for (j=0; j<N; j++)
	    b[j] = scalar*c[j];
}

void tuned_STREAM_Add()
{
	int j;
#pragma omp parallel for
	for (j=0; j<N; j++)
	    c[j] = a[j]+b[j];
}

void tuned_STREAM_Triad(double scalar)
{
	int j;
#pragma omp parallel for
	for (j=0; j<N; j++)
	    a[j] = b[j]+scalar*c[j];
}
