{
	description = "Python dev shell for numerical methods";

	inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
	inputs.flake-utils.url = "github:numtide/flake-utils";

	outputs = { self, nixpkgs, flake-utils }:
		flake-utils.lib.eachDefaultSystem (system:
			let
				pkgs = import nixpkgs { inherit system; };
				pythonEnv = pkgs.python311.withPackages (ps: [
					ps.numpy
					ps.matplotlib
					ps.pandas
					ps.scipy
				]);
			in {
				devShells.default = pkgs.mkShell {
					packages = [ pythonEnv ];
					shellHook = ''
						exec fish
						echo "🐍 $(python --version)"
					'';
				};
			}
		);
}
