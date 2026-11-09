#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>
#include <errno.h>

int main() {
    const char* paths[] = {"/dev/dri/renderD128", "/dev/dri/renderD129",
                           "/dev/dri/card0", "/dev/dri/card1", NULL};
    for (int i = 0; paths[i]; ++i) {
        int fd = open(paths[i], O_RDWR | O_CLOEXEC);
        if (fd < 0) { printf("%-22s open failed: %s\n", paths[i], strerror(errno)); continue; }
        uint64_t has_dumb = 0;
        int cap = drmGetCap(fd, DRM_CAP_DUMB_BUFFER, &has_dumb);
        printf("%-22s open ok fd=%d cap=%d dumb=%llu\n", paths[i], fd, cap,
               (unsigned long long)has_dumb);
        if (cap != 0 || !has_dumb) { close(fd); continue; }

        struct drm_mode_create_dumb cr = {};
        cr.width = 640; cr.height = 640; cr.bpp = 24;
        if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &cr) < 0) {
            printf("    CREATE_DUMB failed: %s\n", strerror(errno)); close(fd); continue;
        }
        struct drm_prime_handle pr = {};
        pr.handle = cr.handle; pr.flags = DRM_CLOEXEC | O_RDWR;
        int r = drmIoctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &pr);
        printf("    CREATE ok handle=%u size=%u pitch=%u  PRIME=%d fd=%d\n",
               cr.handle, cr.size, cr.pitch, r, pr.fd);

        struct drm_mode_map_dumb mr = {};
        mr.handle = cr.handle;
        if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &mr) < 0) {
            printf("    MAP_DUMB failed: %s\n", strerror(errno));
        } else {
            void* p = mmap(NULL, cr.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mr.offset);
            printf("    MAP ok offset=%llu mmap=%s\n", (unsigned long long)mr.offset,
                   p == MAP_FAILED ? strerror(errno) : "ok");
            if (p != MAP_FAILED) munmap(p, cr.size);
        }
        if (pr.fd > 0) close(pr.fd);
        struct drm_mode_destroy_dumb d = {}; d.handle = cr.handle;
        ioctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
        close(fd);
    }
    return 0;
}
