package br.unb.cic.js.miner;

import br.unb.cic.js.miner.JavaScriptParser.*;
import lombok.Getter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

@Getter
public class JSVisitor extends JavaScriptParserBaseVisitor<Void> {

	private static final String THEN = "then";
	private static final String ALL = "all";
	private static final String PROMISE = "Promise";
	private static final String NUMERIC_SEPARATOR = "_";
	private String fileName = "example.js";
	
	public enum Feature {
			ArrowArrowDeclarations,
			AsyncDeclarations,
			AwaitDeclarations,
			LetDeclarations,
			ConstDeclaration,
			ClassDeclarations,
			YieldDeclarations,
			ExportDeclarations,
			ImportStatements,
			RestStatements,
			NewPromises,
			PromiseAllAndThenIdiom,
			ArrayDestructuring,
			ObjectDestructuring,
			DefaultParameters,
			SpreadArguments,
			OptionalChain,
			TemplateStringExpressions,
			RegularExpressions,
			NullCoalesceOperators,
			HashBangLines,
			ExponentiationAssignments,
			PrivateFields,
			NumericLiteralSeparators,
			BigInt,
			ComputedPropertyAssignments,
			EnhancedPropertyAssignments,
			FunctionPropertyDeclarations,
			Statements
	}
	
	private HashMap<Feature, Set<String>> featureOccurrences;
	
	public JSVisitor() {
		super();
		featureOccurrences = new HashMap<>();
		for (Feature f : Feature.values()) {
			featureOccurrences.put(f, new HashSet<>());
		}
	}
	
	public void setFile(String fileName) {
		this.fileName = fileName;
	}
	
	public int occurrences(Feature f) {
		int res = 0;
		if(featureOccurrences.containsKey(f)) {
			res = featureOccurrences.get(f).size();
		}
		return res;
	}
	

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
	AtomicInteger totalOptionalChain = new AtomicInteger(0);
	AtomicInteger totalTemplateStringExpressions = new AtomicInteger(0);
	AtomicInteger totalRegularExpressions = new AtomicInteger(0);
	AtomicInteger totalNullCoalesceOperators = new AtomicInteger(0);
	AtomicInteger totalHashBangLines = new AtomicInteger(0);
	AtomicInteger totalExponentiationAssignments = new AtomicInteger(0);
	AtomicInteger totalPrivateFields = new AtomicInteger(0);
	AtomicInteger totalNumericLiteralSeparators = new AtomicInteger(0);
	AtomicInteger totalBigInt = new AtomicInteger(0);
	AtomicInteger totalComputedPropertyAssignments = new AtomicInteger(0);
	AtomicInteger totalEnhancedPropertyAssignments = new AtomicInteger(0);
	AtomicInteger totalFunctionPropertyDeclarations = new AtomicInteger(0);
	AtomicInteger totalStatements = new AtomicInteger(0);
	
	
	private void changeFilesOccurrences(Feature f) {
		Set<String> files = featureOccurrences.getOrDefault(f, new HashSet<>());
		files.add(fileName);
		featureOccurrences.put(f, files);
	}

	@Override
	public Void visitStatement(StatementContext ctx) {
		totalStatements.incrementAndGet();
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
			changeFilesOccurrences(Feature.AsyncDeclarations);
		}

		return super.visitFunctionDeclaration(ctx);
	}

	@Override
	public Void visitArrowFunction(ArrowFunctionContext ctx) {
		totalArrowDeclarations.incrementAndGet();
		changeFilesOccurrences(Feature.ArrowArrowDeclarations);
		if (ctx.Async() != null) {
			totalAsyncDeclarations.incrementAndGet();
			changeFilesOccurrences(Feature.AsyncDeclarations);
		}
		return super.visitArrowFunction(ctx);
	}

	@Override
	public Void visitAnonymousFunctionDecl(AnonymousFunctionDeclContext ctx) {
		if (ctx.Async() != null) {
			totalAsyncDeclarations.incrementAndGet();
			changeFilesOccurrences(Feature.AsyncDeclarations);
		}
		return super.visitAnonymousFunctionDecl(ctx);
	}

	@Override
	public Void visitFunctionProperty(FunctionPropertyContext ctx) {
		if (ctx.Async() != null) {
			totalAsyncDeclarations.incrementAndGet();
			changeFilesOccurrences(Feature.AsyncDeclarations);
		}
		totalFunctionPropertyDeclarations.incrementAndGet();
		changeFilesOccurrences(Feature.FunctionPropertyDeclarations);
		return super.visitFunctionProperty(ctx);
	}

	@Override
	public Void visitYieldExpression(YieldExpressionContext ctx) {
		if (ctx.yieldStatement() != null) {
			totalYieldDeclarations.incrementAndGet();
			changeFilesOccurrences(Feature.YieldDeclarations);
		}
		return super.visitYieldExpression(ctx);
	}

	@Override
	public Void visitClassDeclaration(ClassDeclarationContext ctx) {
		if (ctx.Class() != null) {
			totalClassDeclarations.incrementAndGet();
			changeFilesOccurrences(Feature.ClassDeclarations);
		}
		return super.visitClassDeclaration(ctx);
	}

	@Override
	public Void visitExportDeclaration(ExportDeclarationContext ctx) {
		if (ctx.Export() != null) {
			totalExportDeclarations.incrementAndGet();
			changeFilesOccurrences(Feature.ExportDeclarations);
		}
		return super.visitExportDeclaration(ctx);
	}
	

	@Override
	public Void visitImportExpression(ImportExpressionContext ctx) {
		if (ctx.Import() != null) {
			totalImportStatements.incrementAndGet();
			changeFilesOccurrences(Feature.ImportStatements);
		}
		return super.visitImportExpression(ctx);
	}

	@Override
	public Void visitImportStatement(ImportStatementContext ctx) {
		if (ctx.Import() != null) {
			totalImportStatements.incrementAndGet();
			changeFilesOccurrences(Feature.ImportStatements);
		}
		return super.visitImportStatement(ctx);
	}

	@Override
	public Void visitLastFormalParameterArg(LastFormalParameterArgContext ctx) {
		if (ctx.Ellipsis() != null) {
			totalRestStatements.incrementAndGet();
			changeFilesOccurrences(Feature.RestStatements);
		}
		return super.visitLastFormalParameterArg(ctx);
	}

	@Override
	public Void visitArrayElement(ArrayElementContext ctx) {
		if (ctx.Ellipsis() != null) {
			totalSpreadArguments.incrementAndGet();
			changeFilesOccurrences(Feature.SpreadArguments);
		}
		return super.visitArrayElement(ctx);
	}

	@Override
	public Void visitPropertyShorthand(PropertyShorthandContext ctx) {
		if (ctx.Ellipsis() != null) {
			totalSpreadArguments.incrementAndGet();
			changeFilesOccurrences(Feature.SpreadArguments);
		}else{
			totalEnhancedPropertyAssignments.incrementAndGet();
			changeFilesOccurrences(Feature.EnhancedPropertyAssignments);
		}
		return super.visitPropertyShorthand(ctx);
	}

	@Override
	public Void visitArgument(ArgumentContext ctx) {
		if (ctx.Ellipsis() != null) {
			totalSpreadArguments.incrementAndGet();
			changeFilesOccurrences(Feature.SpreadArguments);
		}
		return super.visitArgument(ctx);
	}

	@Override
	public Void visitVarModifier(VarModifierContext ctx) {
		if (ctx.Const() != null) {
			totalConstDeclaration.incrementAndGet();
			changeFilesOccurrences(Feature.ConstDeclaration);
		}
		if (ctx.let_() != null) {
			totalLetDeclarations.incrementAndGet();
			changeFilesOccurrences(Feature.LetDeclarations);
		}
		return super.visitVarModifier(ctx);
	}

	@Override
	public Void visitAwaitExpression(AwaitExpressionContext ctx) {
		if (ctx.Await() != null) {
			totalAwaitDeclarations.incrementAndGet();
			changeFilesOccurrences(Feature.AwaitDeclarations);
		}
		return super.visitAwaitExpression(ctx);
	}

	@Override
	public Void visitIdentifier(IdentifierContext ctx) {
		if (ctx.Async() != null) {
			totalAsyncDeclarations.incrementAndGet();
			changeFilesOccurrences(Feature.AsyncDeclarations);
		}
		return super.visitIdentifier(ctx);
	}


	@Override
	public Void visitMethodDefinition(MethodDefinitionContext ctx) {
		if (ctx.Async() != null) {
			totalAsyncDeclarations.incrementAndGet();
			changeFilesOccurrences(Feature.AsyncDeclarations);
		}
		return super.visitMethodDefinition(ctx);
	}

	@Override
	public Void visitNewExpression(NewExpressionContext ctx) {
		if (ctx.singleExpression() != null && ctx.singleExpression().getText().equals(PROMISE)) {
			totalNewPromises.incrementAndGet();
			changeFilesOccurrences(Feature.NewPromises);
		}
		if (ctx.identifier() != null && ctx.identifier().Identifier() != null
				&& ctx.identifier().Identifier().getText().equals(PROMISE)) {
			totalNewPromises.incrementAndGet();
			changeFilesOccurrences(Feature.NewPromises);
		}

		return super.visitNewExpression(ctx);
	}


	@Override
	public Void visitArgumentsExpression(ArgumentsExpressionContext ctx) {
		if (ctx.singleExpression().getText().contains(PROMISE) && ctx.singleExpression().getText().contains(ALL)
				&& ctx.singleExpression().getText().contains(THEN)) {
			totalPromiseAllAndThenIdiom.incrementAndGet();
			changeFilesOccurrences(Feature.PromiseAllAndThenIdiom);
		}
		return super.visitArgumentsExpression(ctx);
	}

	@Override
	public Void visitAssignmentExpression(AssignmentExpressionContext ctx) {
		if (ctx.singleExpression().get(0) instanceof ArrayLiteralExpressionContext) {
			totalArrayDestructuring.incrementAndGet();
			changeFilesOccurrences(Feature.ArrayDestructuring);
		} else if (ctx.singleExpression().get(0) instanceof ObjectLiteralExpressionContext) {
			totalObjectDestructuring.incrementAndGet();
			changeFilesOccurrences(Feature.ObjectDestructuring);
		}
		return super.visitAssignmentExpression(ctx);
	}

	@Override
	public Void visitVariableDeclaration(VariableDeclarationContext ctx) {
		if (ctx.singleExpression() != null && !ctx.assignable().isEmpty()) {
			if (ctx.assignable().arrayLiteral() != null) {
				totalArrayDestructuring.incrementAndGet();
				changeFilesOccurrences(Feature.ArrayDestructuring);
			} else if (ctx.assignable().objectLiteral() != null) {
				totalObjectDestructuring.incrementAndGet();
				changeFilesOccurrences(Feature.ObjectDestructuring);
			}
		}
		return super.visitVariableDeclaration(ctx);
	}

	@Override
	public Void visitFormalParameterArg(FormalParameterArgContext ctx) {
		if (ctx.singleExpression() != null) {
			totalDefaultParameters.incrementAndGet();
			changeFilesOccurrences(Feature.DefaultParameters);
		}
		return super.visitFormalParameterArg(ctx);
	}

	@Override
	public Void visitOptionalChainExpression(OptionalChainExpressionContext ctx) {
		if (!ctx.isEmpty()) {
			totalOptionalChain.incrementAndGet();
			changeFilesOccurrences(Feature.OptionalChain);
		}
		return super.visitOptionalChainExpression(ctx);
	}

	@Override
	public Void visitTemplateStringLiteral(TemplateStringLiteralContext ctx) {
		if (ctx.templateStringAtom() != null) {
			totalTemplateStringExpressions.incrementAndGet();
			changeFilesOccurrences(Feature.TemplateStringExpressions);
		}
		return super.visitTemplateStringLiteral(ctx);
	}

	@Override
	public Void visitComputedPropertyExpressionAssignment(ComputedPropertyExpressionAssignmentContext ctx) {
		totalComputedPropertyAssignments.incrementAndGet();
		changeFilesOccurrences(Feature.ComputedPropertyAssignments);
		return super.visitComputedPropertyExpressionAssignment(ctx);
	}

	@Override
	public Void visitLiteral(LiteralContext ctx) {
		if (ctx.RegularExpressionLiteral() != null) {
			String regexText = ctx.RegularExpressionLiteral().getText();
			// Verifica se a expressão regular contém expressões nomeadas ou lookbehind
			if (regexText.contains("?<") || regexText.contains("(?<=")) {
				totalRegularExpressions.incrementAndGet();
				changeFilesOccurrences(Feature.RegularExpressions);
			}
		}
		return super.visitLiteral(ctx);
	}

	@Override
	public Void visitCoalesceExpression(CoalesceExpressionContext ctx) {
		if (ctx.NullCoalesce() != null) {
			totalNullCoalesceOperators.incrementAndGet();
			changeFilesOccurrences(Feature.NullCoalesceOperators);
		}
		return super.visitCoalesceExpression(ctx);
	}

	@Override
	public Void visitForOfStatement(ForOfStatementContext ctx) {
		if (ctx.Await() != null) {
			totalAwaitDeclarations.incrementAndGet();
			changeFilesOccurrences(Feature.AwaitDeclarations);
		}
		return super.visitForOfStatement(ctx);
	}

	@Override
	public Void visitProgram(ProgramContext ctx) {

		if (ctx.HashBangLine() != null) {
			totalHashBangLines.incrementAndGet();
			changeFilesOccurrences(Feature.HashBangLines);
		}

		return super.visitProgram(ctx);
	}

	@Override
	public Void visitAssignmentOperator(AssignmentOperatorContext ctx) {
		if (ctx.PowerAssign()!=null) {
			totalExponentiationAssignments.incrementAndGet();
			changeFilesOccurrences(Feature.ExponentiationAssignments);
		}
		return super.visitAssignmentOperator(ctx);
	}

	@Override
	public Void visitFieldDefinition(FieldDefinitionContext ctx) {
		if (ctx.classElementName().privateIdentifier() != null) {
			totalPrivateFields.incrementAndGet();
			changeFilesOccurrences(Feature.PrivateFields);
		}
		return super.visitFieldDefinition(ctx);
	}

	@Override
	public Void visitLiteralExpression(LiteralExpressionContext ctx) {
		
		if (ctx.literal().numericLiteral() != null && ctx.literal().getText().contains(NUMERIC_SEPARATOR)) {
			totalNumericLiteralSeparators.incrementAndGet();
			changeFilesOccurrences(Feature.NumericLiteralSeparators);
		}
		return super.visitLiteralExpression(ctx);
	}

	@Override
	public Void visitBigintLiteral(BigintLiteralContext ctx) {
		if (!ctx.isEmpty()) {
			totalBigInt.incrementAndGet();
			changeFilesOccurrences(Feature.BigInt);
		}
		return super.visitBigintLiteral(ctx);
	}

}
