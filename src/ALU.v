module ALU (
    input [3:0] ALUCtl,
    input [31:0] A,B,
    output reg [31:0] ALUOut,
    output zero
);
    // Zero is set if ALUOut is zero
    assign zero = (ALUOut == 0);

    always @(*) begin
        case (ALUCtl)
            4'b0000: ALUOut = A & B;         // AND
            4'b0001: ALUOut = A | B;         // OR
            4'b0010: ALUOut = A + B;         // ADD
            4'b0110: ALUOut = A - B;         // SUB
            4'b0111: ALUOut = (A < B) ? 1 : 0; // SLT
            4'b1100: ALUOut = ~(A | B);      // NOR
            default: ALUOut = 32'b0;         // Default case
        endcase
    end
    
endmodule

