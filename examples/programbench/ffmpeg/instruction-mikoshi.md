Recreate FFmpeg from scratch using its observable behavior and the bundled
documentation. Work in /workspace. The query-only reference executable is
/opt/programbench-reference/executable, available in every command container.
Run it with CLI arguments, files, and stdin to learn the required behavior.
README.md, the license files, and assets/ are the original cleanroom materials.

Write an independent implementation. Do not obtain or use the original source,
install an existing FFmpeg implementation, wrap or delegate to the reference,
reuse prebuilt FFmpeg objects, or decompile, disassemble, trace, or instrument the
reference. Only its normal user interface and the bundled documentation may
inform your implementation. Network access is available for build dependencies
and general tooling. Preserve the independent-implementation rules when using
it. Analysis tools may be used on binaries you produce yourself.

Provide an executable ./compile.sh that builds a new ./executable from your
source in a fresh, offline cleanroom. Vendor any source dependencies you download
that are not already in the cleanroom. The grader deletes any pre-existing
./executable before building. Your implementation must run without the reference
or an installed copy of FFmpeg. All required source and local dependencies must
be in the submission or already present in the cleanroom. Keep generated
executables and build artifacts out of Git. Test your implementation, then commit
your source changes. Mikoshi's normal completion tools end the task.
