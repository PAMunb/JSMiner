package br.unb.cic.js.walker;

import br.unb.cic.js.walker.exception.NoBranchFoundException;
import lombok.NoArgsConstructor;
import lombok.val;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.ListBranchCommand;
import org.eclipse.jgit.api.ListBranchCommand.ListMode;
import org.eclipse.jgit.api.ResetCommand;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Ref;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.revwalk.RevCommit;
import org.eclipse.jgit.revwalk.RevWalk;

@NoArgsConstructor
final class RepositoryWalkerGit {

	private static List<String> removerDuplicates(List<String> lista) {
		Set<String> conjuntoSemRepeticoes = new LinkedHashSet<>(lista);
		return new ArrayList<>(conjuntoSemRepeticoes);
	}

	public static ObjectId head(final Repository repository) throws Exception {
		var mainBranch = "";

		try (val git = new Git(repository)) {
			val branches = git.branchList().setListMode(ListBranchCommand.ListMode.REMOTE).call().stream()
					.filter(n -> n.getName().equals("refs/remotes/origin/HEAD")).findFirst();

			if (branches.isPresent()) {
				mainBranch = branches.get().getTarget().getName().substring("refs/remotes/origin/".length());
			} else {
				throw new NoBranchFoundException();
			}

			git.reset().setMode(ResetCommand.ResetType.HARD).call();
			git.checkout().setName(mainBranch).call();
		}

		return repository.resolve("refs/heads/" + mainBranch);
	}

	// Method to get a list of commits in linear order from a Git repository
	public static Iterable<RevCommit> revisions(Repository repository, Boolean merges) throws Exception {
		try (Git git = new Git(repository)) {
			String latestBranchName = getBranchNameFromLatestCommit(repository, git);
			// List<String> targetBranches = removerDuplicates(
			// 		Arrays.asList("main", "master", latestBranchName.substring(latestBranchName.lastIndexOf("/") + 1)));
			List<String> targetBranches = removerDuplicates(
				Arrays.asList(latestBranchName.substring(latestBranchName.lastIndexOf("/") + 1)));

			boolean startsWithAny = false;

			for (String branch : targetBranches) {

				for (Ref branchRef : git.branchList().setListMode(ListMode.REMOTE).call()) {
					String branchName = Repository.shortenRefName(branchRef.getName());

					if (branchName.substring(branchName.lastIndexOf("/") + 1).equals(branch)) {
						startsWithAny = true;
					}

					if (startsWithAny) {
						ObjectId branchObjectId = repository.resolve(branchName);
						List<RevCommit> commits = listCommits(repository, branchObjectId, merges);
						return commits;
					} else {
						continue;
					}
				}

				if (startsWithAny) {
					break;
				}

			}

		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	private static List<RevCommit> listCommits(Repository repository, ObjectId branchObjectId, Boolean merges)
			throws Exception {
		try (RevWalk revWalk = new RevWalk(repository)) {
			revWalk.markStart(revWalk.parseCommit(branchObjectId));

			if (merges == false) {
				return StreamSupport.stream(revWalk.spliterator(), false).filter(r -> r.getParentCount() == 1)
						.collect(Collectors.toList());
			} else {
				return StreamSupport.stream(revWalk.spliterator(), false).collect(Collectors.toList());
			}
		}
	}
	
	private static String getBranchNameFromLatestCommit(Repository repository, Git git) throws Exception {
		try (RevWalk revWalk = new RevWalk(repository)) {
			var mainBranch = "";
			val branches = git.branchList().setListMode(ListBranchCommand.ListMode.REMOTE).call().stream()
					.filter(n -> n.getName().equals("refs/remotes/origin/HEAD")).findFirst();

			if (branches.isPresent()) {
				mainBranch = branches.get().getTarget().getName().substring("refs/remotes/origin/".length());
				return mainBranch;
			} else {
				throw new Exception();
			}
		}
	}
}