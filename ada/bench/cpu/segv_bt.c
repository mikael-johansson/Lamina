// LD_PRELOAD=.../libsegv_bt.so (gcc -O1 -shared -fPIC segv_bt.c -o libsegv_bt.so): on SIGSEGV/SIGBUS/SIGABRT, the faulting thread's backtrace to stderr (and to
// $SEGV_BT_OUT when set), then the default action.  For a crash in a process a server starts.
#define _GNU_SOURCE
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>
static void handler(int sig, siginfo_t* si, void* ctx) {
    void* frames[64];
    int n = backtrace(frames, 64);
    ucontext_t* uc = (ucontext_t*) ctx;
    void* pc = (void*) uc->uc_mcontext.gregs[REG_RIP];
    const char* out = getenv("SEGV_BT_OUT");
    int fds[2] = {2, out ? open(out, O_WRONLY | O_CREAT | O_APPEND, 0644) : -1};
    for (int i = 0; i < 2; ++i) {
        if (fds[i] < 0) continue;
        char head[160];
        int len = 0;
        len += snprintf(head, sizeof head, "segv_bt: signal %d at address %p, pc %p, thread %d\n", sig, si->si_addr, pc,
                        (int) gettid());
        write(fds[i], head, len);
        backtrace_symbols_fd(&pc, 1, fds[i]);
        backtrace_symbols_fd(frames, n, fds[i]);
    }
    signal(sig, SIG_DFL);
    raise(sig);
}
__attribute__((constructor)) static void init(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    static char stack[1 << 16];
    stack_t ss = {.ss_sp = stack, .ss_size = sizeof stack};
    sigaltstack(&ss, 0);
    sigaction(SIGSEGV, &sa, 0);
    sigaction(SIGBUS, &sa, 0);
    sigaction(SIGABRT, &sa, 0);
}
