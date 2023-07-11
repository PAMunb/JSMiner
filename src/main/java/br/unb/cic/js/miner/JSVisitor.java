package br.unb.cic.js.miner;

import br.unb.cic.js.miner.JavaScriptParser.*;
import lombok.Getter;
import lombok.val;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;

@Getter
public class JSVisitor extends JavaScriptParserBaseVisitor<Void> {

	private static final String THEN = "then";
	private static final String ALL = "all";
	private static final String PROMISE = "Promise";

	AtomicInteger totalArrowDeclarations = new AtomicInteger(0);
	AtomicInteger totalAsyncDeclarations = new AtomicInteger(0);
	AtomicInteger totalAwaitDeclarations = new AtomicInteger(0);
	AtomicInteger totalLetDeclarations = new AtomicInteger(0);
	AtomicInteger totalConstDeclaration = new AtomicInteger(0);
	AtomicInteger totalClassDeclarations = new AtomicInteger(0);
	AtomicInteger totalYieldDeclarations = new AtomicInteger(0);
	AtomicInteger totalExportDeclarations = new AtomicInteger(0);
	AtomicInteger totalImportStatements = new AtomicInteger(0);
	AtomicInteger totalRestStatements = new AtomicInteger(0);
	AtomicInteger totalNewPromises = new AtomicInteger(0);
	AtomicInteger totalPromiseAllAndThenIdiom = new AtomicInteger(0);
	AtomicInteger totalArrayDestructuring = new AtomicInteger(0);
	AtomicInteger totalObjectDestructuring = new AtomicInteger(0);
	AtomicInteger totalDefaultParameters = new AtomicInteger(0);
	AtomicInteger totalSpreadArguments = new AtomicInteger(0);
	AtomicInteger totalStatements = new AtomicInteger(0);

	@Override
	public Void visitStatement(StatementContext ctx) {
		val statements = new ArrayList<Boolean>();

		statements.add(ctx.variableStatement() != null);
		statements.add(ctx.importStatement() != null);
		statements.add(ctx.importStatement() != null);
		statements.add(ctx.block() != null);
		statements.add(ctx.exportStatement() != null);
		statements.add(ctx.emptyStatement_() != null);
		statements.add(ctx.classDeclaration() != null);
		statements.add(ctx.expressionStatement() != null);
		statements.add(ctx.ifStatement() != null);
		statements.add(ctx.iterationStatement() != null);
		statements.add(ctx.continueStatement() != null);
		statements.add(ctx.breakStatement() != null);
		statements.add(ctx.returnStatement() != null);
		statements.add(ctx.yieldStatement() != null);
		statements.add(ctx.withStatement() != null);
		statements.add(ctx.labelledStatement() != null);
		statements.add(ctx.switchStatement()!= null);
		statements.add(ctx.throwStatement() != null);
		statements.add(ctx.tryStatement() != null);
		statements.add(ctx.debuggerStatement() != null);
		statements.add(ctx.functionDeclaration() != null);

		if (statements.stream().reduce((m, n) -> m || n).orElse(false)) {
			totalStatements.incrementAndGet();
		}

		return super.visitStatement(ctx);
	}

	@Override
	public Void visitStatementList(StatementListContext ctx) {
		if (ctx.statement() != null) {
			totalStatements.incrementAndGet();
		}
		return super.visitStatementList(ctx);
	}

	@Override
	public Void visitFunctionDeclaration(FunctionDeclarationContext ctx) {
		if (ctx.Async() != null) {
			totalAsyncDeclarations.incrementAndGet();
		}
		return super.visitFunctionDeclaration(ctx);
	}

	@Override
	public Void visitArrowFunction(ArrowFunctionContext ctx) {
		totalArrowDeclarations.incrementAndGet();
		if (ctx.Async() != null) {
			totalAsyncDeclarations.incrementAndGet();
		}
		return super.visitArrowFunction(ctx);
	}

	@Override
	public Void visitAnonymousFunctionDecl(AnonymousFunctionDeclContext ctx) {
		totalArrowDeclarations.incrementAndGet();
		if (ctx.Async() != null) {
			totalAsyncDeclarations.incrementAndGet();
		}
		return super.visitAnonymousFunctionDecl(ctx);
	}

	@Override
	public Void visitFunctionProperty(FunctionPropertyContext ctx) {
		if (ctx.Async() != null) {
			totalAsyncDeclarations.incrementAndGet();
		}
		return super.visitFunctionProperty(ctx);
	}

	@Override
	public Void visitYieldExpression(YieldExpressionContext ctx) {
		if (ctx.yieldStatement() != null) {
			totalYieldDeclarations.incrementAndGet();
		}
		return super.visitYieldExpression(ctx);
	}

	@Override
	public Void visitClassDeclaration(ClassDeclarationContext ctx) {
		if (ctx.Class() != null) {
			totalClassDeclarations.incrementAndGet();
		}
		return super.visitClassDeclaration(ctx);
	}

	@Override
	public Void visitExportDeclaration(ExportDeclarationContext ctx) {
		if (ctx.Export() != null) {
			totalExportDeclarations.incrementAndGet();
		}
		return super.visitExportDeclaration(ctx);
	}

	@Override
	public Void visitImportExpression(ImportExpressionContext ctx) {
		if (ctx.Import() != null) {
			totalImportStatements.incrementAndGet();
		}
		return super.visitImportExpression(ctx);
	}

	@Override
	public Void visitImportStatement(ImportStatementContext ctx) {
		if (ctx.Import() != null) {
			totalImportStatements.incrementAndGet();
		}
		return super.visitImportStatement(ctx);
	}

	@Override
	public Void visitLastFormalParameterArg(LastFormalParameterArgContext ctx) {
		if (ctx.Ellipsis() != null) {
			totalRestStatements.incrementAndGet();
		}
		return super.visitLastFormalParameterArg(ctx);
	}

	@Override
	public Void visitVarModifier(VarModifierContext ctx) {
		if (ctx.Const() != null) {
			totalConstDeclaration.incrementAndGet();
		}
		if (ctx.let_() != null) {
			totalLetDeclarations.incrementAndGet();
		}
		return super.visitVarModifier(ctx);
	}

	@Override
	public Void visitAwaitExpression(AwaitExpressionContext ctx) {
		if (ctx.Await() != null) {
			totalAwaitDeclarations.incrementAndGet();
		}
		return super.visitAwaitExpression(ctx);
	}

	@Override
	public Void visitIdentifier(IdentifierContext ctx) {
		if (ctx.Async() != null) {
			totalAsyncDeclarations.incrementAndGet();
		}
		return super.visitIdentifier(ctx);
	}

	@Override
	public Void visitClassElementMethodDefinition(ClassElementMethodDefinitionContext ctx) {
		if(ctx.Async() != null) {
			totalAsyncDeclarations.incrementAndGet();
		}
		return super.visitClassElementMethodDefinition(ctx);
	}

	@Override
	public Void visitClassElementAssigment(ClassElementAssigmentContext ctx) {
		if(ctx.Async() != null) {
			totalAsyncDeclarations.incrementAndGet();
		}
		return super.visitClassElementAssigment(ctx);
	}

	@Override
	public Void visitNewExpression(NewExpressionContext ctx) {
		if (ctx.singleExpression().getText().equals(PROMISE)) {
			totalNewPromises.incrementAndGet();
		}
		return super.visitNewExpression(ctx);
	}

	@Override
	public Void visitArgumentsExpression(ArgumentsExpressionContext ctx) {
		if (ctx.singleExpression().getText().contains(PROMISE) &&
				ctx.singleExpression().getText().contains(ALL) &&
				ctx.singleExpression().getText().contains(THEN)) {
			totalPromiseAllAndThenIdiom.incrementAndGet();
		}
		return super.visitArgumentsExpression(ctx);
	}

	@Override
	public Void visitAssignmentExpression(AssignmentExpressionContext ctx) {
		if (ctx.singleExpression().get(0) instanceof ArrayLiteralExpressionContext) {
			totalArrayDestructuring.incrementAndGet();
		}else if(ctx.singleExpression().get(0) instanceof ObjectLiteralExpressionContext) {
			totalObjectDestructuring.incrementAndGet();
		}
		return super.visitAssignmentExpression(ctx);
	}

	@Override
	public Void visitVariableDeclaration(VariableDeclarationContext ctx) {
		if(ctx.singleExpression() != null && !ctx.assignable().isEmpty()) {
			if (ctx.assignable().arrayLiteral() != null) {
				totalArrayDestructuring.incrementAndGet();
			}
			else if (ctx.assignable().objectLiteral() != null) {
				totalObjectDestructuring.incrementAndGet();
			}
		}
		return super.visitVariableDeclaration(ctx);
	}

	@Override
	public Void visitFormalParameterArg(FormalParameterArgContext ctx) {
		if (ctx.singleExpression() != null) {
			totalDefaultParameters.incrementAndGet();
		}
		return super.visitFormalParameterArg(ctx);
	}

	@Override
	public Void visitArgument(ArgumentContext ctx) {
		if (ctx.Ellipsis() != null) {
			totalSpreadArguments.incrementAndGet();
		}
		return super.visitArgument(ctx);
	}
}
