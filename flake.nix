{
  description = "polars-psf: Cadence Spectre PSF reader (Rust + Python/Polars) dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rustc
            clippy
            rustfmt
            rust-analyzer
            python3
            uv
            maturin
          ];
          # manylinux wheels installed by uv/pip (numpy, polars, pyarrow) expect these at runtime
          LD_LIBRARY_PATH = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux
            (pkgs.lib.makeLibraryPath [ pkgs.zlib pkgs.stdenv.cc.cc.lib ]);
          shellHook = ''
            echo "polars-psf dev shell: cargo test | cd python && maturin build --release -o dist"
          '';
        };
      });
    };
}
