// LD_PRELOAD shim for the CPU-only Isaac physics worker (sim/physics_client.py nosmi=True, docs/throughput.md 8.4).
// With CUDA and Vulkan hidden, Kit's gpu.foundation (SystemValidation launchNvidiaSmi, through carb.launcher:
// fork + exec*) runs nvidia-smi on every stage attach and timeline stop (two per episode) to explain the missing GPU.
// nvidia-smi serialises in the driver: 2.5-3 calls/s per node whatever the concurrency (measured), which caps a node
// at ~1.3 episodes/s. Here an exec whose path or arguments name nvidia-smi runs /bin/true instead, so the probe gets
// no output and fails at once, as it fails anyway. Nothing else is touched.
// Build: gcc -O2 -shared -fPIC -o libnosmi.so nosmi.c -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int names_smi(const char *path, char *const argv[]) {
    if (path && strstr(path, "nvidia-smi")) return 1;
    for (int i = 0; argv && argv[i]; i++)
        if (strstr(argv[i], "nvidia-smi")) return 1;
    return 0;
}

static char *const TRUE_ARGV[] = {"true", NULL};
typedef int (*execve_t)(const char *, char *const[], char *const[]);
typedef int (*execv_t)(const char *, char *const[]);

static int run_true(char *const envp[]) {
    static const char msg[] = "[nosmi] nvidia-smi exec replaced by /bin/true\n";
    if (write(2, msg, sizeof msg - 1) < 0) {}
    return ((execve_t)dlsym(RTLD_NEXT, "execve"))("/bin/true", TRUE_ARGV, envp);
}

int execve(const char *path, char *const argv[], char *const envp[]) {
    if (names_smi(path, argv)) return run_true(envp);
    return ((execve_t)dlsym(RTLD_NEXT, "execve"))(path, argv, envp);
}
int execv(const char *path, char *const argv[]) {
    if (names_smi(path, argv)) return run_true(environ);
    return ((execv_t)dlsym(RTLD_NEXT, "execv"))(path, argv);
}
int execvp(const char *file, char *const argv[]) {
    if (names_smi(file, argv)) return run_true(environ);
    return ((execv_t)dlsym(RTLD_NEXT, "execvp"))(file, argv);
}
int execvpe(const char *file, char *const argv[], char *const envp[]) {
    if (names_smi(file, argv)) return run_true(envp);
    return ((execve_t)dlsym(RTLD_NEXT, "execvpe"))(file, argv, envp);
}
FILE *popen(const char *command, const char *type) {
    FILE *(*real)(const char *, const char *) = (FILE * (*)(const char *, const char *)) dlsym(RTLD_NEXT, "popen");
    if (command && strstr(command, "nvidia-smi")) return real("true", type);
    return real(command, type);
}
