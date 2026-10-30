{
  description = "Polylogue";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.rust-overlay = {
    url = "github:oxalica/rust-overlay/4748ec2f5ed4a881474ed4c98aa71a5308cdac8d";
    inputs.nixpkgs.follows = "nixpkgs";
  };
  inputs.crane.url = "github:ipetkov/crane/eb35abda9f232cc6610b1d1e3200d15c49b7ac54";
  inputs.home-manager = {
    url = "github:nix-community/home-manager/2c0350c759688177331b8f5242311fae8877bdb3";
    inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      crane,
      home-manager,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };
      toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      build = import ./nix/build.nix {
        inherit pkgs toolchain crane;
        source = self.outPath;
        revision = self.rev or self.dirtyRev or "unavailable";
      };
    in
    {
      packages.${system} = build.packages;
      checks.${system} = build.checks // {
        modules = import ./nix/checks/modules.nix {
          inherit pkgs nixpkgs;
          homeManager = home-manager;
          inherit (self) homeModules nixosModules;
        };
        services = import ./nix/checks/services.nix {
          inherit pkgs;
          homeManager = home-manager;
          homeModule = self.homeModules.polylogue;
          polylogue = build.packages.polylogue;
        };
      };
      homeModules.polylogue = import ./nix/hm-module.nix {
        packageFor = target: self.packages.${target};
      };
      nixosModules.polylogue = import ./nix/module.nix {
        packageFor = target: self.packages.${target};
      };
      formatter.${system} = pkgs.nixfmt-tree;
      apps.${system} = {
        polylogue = {
          type = "app";
          program = "${build.packages.polylogue}/bin/polylogue";
        };
        default = self.apps.${system}.polylogue;
      };
      overlays.default = final: _: {
        inherit (self.packages.${final.stdenv.hostPlatform.system}) polylogue polylogue-extension;
      };
      devShells.${system}.default = build.devShell;
    };
}
