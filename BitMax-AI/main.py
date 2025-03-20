from fastapi import FastAPI, HTTPException, Depends, Body, status
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from contextlib import asynccontextmanager

# Import the AI agent
from yield_tokenization_agent import YieldTokenizationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - initialize any resources
    logger.info("Starting PT/YT ROI Optimization API")
    yield
    # Shutdown - clean up resources
    logger.info("Shutting down PT/YT ROI Optimization API")

app = FastAPI(
    title="PT/YT ROI Optimization API",
    description="API for optimizing ROI with Principal Tokens (PT) and Yield Tokens (YT)",
    version="1.0.0",
    lifespan=lifespan
)

# Define data models for request and response
class TokenPosition(BaseModel):
    token_type: str  # PT or YT
    asset: str  # BTC or CORE
    amount: float
    value_usd: float
    maturity_date: Optional[str] = None
    
    @field_validator('token_type')
    def validate_token_type(cls, v):
        if v not in ["PT", "YT"]:
            raise ValueError('token_type must be PT or YT')
        return v
    
    @field_validator('asset')
    def validate_asset(cls, v):
        if v not in ["BTC", "CORE"]:
            raise ValueError('asset must be BTC or CORE')
        return v
    
    @field_validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('amount must be positive')
        return v

class UserProfile(BaseModel):
    risk_tolerance: str = Field(..., description="User's risk tolerance (low, medium, high)")
    investment_horizon: str = Field(..., description="User's investment horizon (short, medium, long)")
    financial_goal: str = Field(..., description="User's primary financial goal")
    
    @field_validator('risk_tolerance')
    def validate_risk_tolerance(cls, v):
        if v.lower() not in ["low", "medium", "high"]:
            raise ValueError('risk_tolerance must be low, medium, or high')
        return v.lower()
    
    @field_validator('investment_horizon')
    def validate_investment_horizon(cls, v):
        if v.lower() not in ["short", "medium", "long"]:
            raise ValueError('investment_horizon must be short, medium, or long')
        return v.lower()

class MarketData(BaseModel):
    btc_yield: float = Field(..., ge=0, description="Bitcoin yield rate")
    core_yield: float = Field(..., ge=0, description="CORE yield rate")
    pt_btc_price: float = Field(..., gt=0, lt=1, description="PT-BTC price (fraction of face value)")
    pt_core_price: float = Field(..., gt=0, lt=1, description="PT-CORE price (fraction of face value)")
    yt_btc_price: float = Field(..., gt=0, description="YT-BTC price")
    yt_core_price: float = Field(..., gt=0, description="YT-CORE price")
    available_maturities: List[str] = Field(..., description="Available maturity dates")

class UserPortfolio(BaseModel):
    positions: List[TokenPosition]
    total_value_usd: float
    
    @field_validator('total_value_usd')
    def validate_total_value(cls, v, values):
        positions = values.data.get('positions', [])
        calculated_total = sum(pos.value_usd for pos in positions)
        if abs(v - calculated_total) > 0.01:  # Allow small floating point differences
            raise ValueError(f'total_value_usd ({v}) does not match sum of positions ({calculated_total})')
        return v

class OptimizationRequest(BaseModel):
    user_profile: UserProfile
    market_data: Optional[MarketData] = None
    portfolio: Optional[UserPortfolio] = None

class Action(BaseModel):
    action: str = Field(..., description="Action to take (buy, sell, hold)")  
    token: str = Field(..., description="Token identifier (PT-BTC, YT-CORE, etc.)")
    percentage: float = Field(..., ge=0, le=100, description="Percentage of holdings to act on")
    
    @field_validator('action')
    def validate_action(cls, v):
        if v.lower() not in ["buy", "sell", "hold"]:
            raise ValueError('action must be buy, sell, or hold')
        return v.lower()
    
    @field_validator('token')
    def validate_token(cls, v):
        valid_tokens = ["PT-BTC", "PT-CORE", "YT-BTC", "YT-CORE"]
        if v not in valid_tokens:
            raise ValueError(f'token must be one of {valid_tokens}')
        return v

class Strategy(BaseModel):
    name: str
    description: str
    actions: List[Action]
    rationale: str
    expected_roi: float
    risk_score: float
    confidence: float

class MarketOutlook(BaseModel):
    btc_yield_trend: str
    core_yield_trend: str
    optimal_timeframe: str
    confidence: float

class SimulationResult(BaseModel):
    initial_value: float
    expected_value: float
    expected_roi: float
    risk_assessment: str
    confidence_interval: List[float]

class PortfolioImpact(BaseModel):
    before: Dict[str, Any]
    after: Dict[str, Any]
    expected_changes: Dict[str, Any]

class OptimizationResponse(BaseModel):
    recommended_strategy: Strategy
    alternative_strategies: List[Strategy]
    market_outlook: MarketOutlook
    simulation_result: SimulationResult
    portfolio_impact: PortfolioImpact

# Market data service (in a real implementation, this would fetch data from APIs)
async def get_market_data():
    # This is a placeholder - in production, you'd fetch real market data
    logger.info("Fetching market data")
    return {
        "btc_yield": 0.045,  # 4.5% annual yield
        "core_yield": 0.078,  # 7.8% annual yield
        "pt_btc_price": 0.965,  # PT tokens trade at slight discount to face value
        "pt_core_price": 0.942,
        "yt_btc_price": 0.035,
        "yt_core_price": 0.062,
        "available_maturities": ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    }

# API endpoints
@app.post("/optimize", response_model=OptimizationResponse, status_code=status.HTTP_200_OK)
async def optimize_portfolio(
    request: OptimizationRequest = Body(...),
    market_data: Dict = Depends(get_market_data)
):
    try:
        logger.info(f"Optimizing portfolio for user with risk tolerance: {request.user_profile.risk_tolerance}")
        
        # Initialize the AI agent
        agent = YieldTokenizationAgent()
        
        # Load market data (either from request or from service)
        if request.market_data:
            logger.info("Using client-provided market data")
            agent.load_market_data(request.market_data.model_dump())
        else:
            logger.info("Using service-provided market data")
            agent.load_market_data(market_data)
        
        # Set user profile
        agent.set_user_profile(request.user_profile.model_dump())
        
        # Register current positions if available
        if request.portfolio:
            logger.info(f"Processing user portfolio with {len(request.portfolio.positions)} positions")
            formatted_positions = [
                {
                    "token_type": pos.token_type,
                    "asset": pos.asset,
                    "amount": pos.amount,
                    "value_usd": pos.value_usd,
                    "maturity_date": pos.maturity_date
                }
                for pos in request.portfolio.positions
            ]
            agent.register_positions(formatted_positions)
        
        # Get recommendation
        logger.info("Generating strategy recommendations")
        recommendation = agent.recommend_strategy()
        
        # Check for errors in recommendation
        if "error" in recommendation:
            logger.error(f"Error in recommendation: {recommendation['error']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Error in recommendation: {recommendation['error']}"
            )
        
        # Simulate the recommended strategy
        logger.info(f"Simulating recommended strategy: {recommendation['recommended']['name']}")
        simulation = agent.simulate_strategy(recommendation["recommended"])
        
        # Calculate portfolio impact
        logger.info("Calculating portfolio impact")
        portfolio_impact = calculate_portfolio_impact(
            recommendation["recommended"], 
            request.portfolio.positions if request.portfolio else []
        )
        
        # Format response
        response = OptimizationResponse(
            recommended_strategy=recommendation["recommended"],
            alternative_strategies=recommendation["alternatives"],
            market_outlook=recommendation["market_outlook"],
            simulation_result=simulation,
            portfolio_impact=portfolio_impact
        )
        
        logger.info("Optimization complete")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error optimizing portfolio: {str(e)}")

@app.post("/explain", status_code=status.HTTP_200_OK)
async def explain_strategy(strategy_name: str, user_profile: UserProfile):
    try:
        logger.info(f"Explaining strategy '{strategy_name}' for user with risk tolerance: {user_profile.risk_tolerance}")
        
        agent = YieldTokenizationAgent()
        agent.set_user_profile(user_profile.model_dump())
        
        # Get market data
        market_data = await get_market_data()
        agent.load_market_data(market_data)
        
        explanation = agent.explain_recommendation(strategy_name)
        return {"explanation": explanation}
        
    except Exception as e:
        logger.error(f"Error explaining strategy: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error explaining strategy: {str(e)}")

@app.post("/simulate", status_code=status.HTTP_200_OK)
async def simulate_strategy(
    strategy: Strategy,
    user_profile: UserProfile,
    time_horizon: str = "3m"
):
    try:
        logger.info(f"Simulating strategy '{strategy.name}' for time horizon '{time_horizon}'")
        
        # Validate time horizon
        valid_horizons = ["1m", "3m", "6m", "1y"]
        if time_horizon not in valid_horizons:
            raise ValueError(f"time_horizon must be one of {valid_horizons}")
        
        agent = YieldTokenizationAgent()
        agent.set_user_profile(user_profile.model_dump())
        
        # Get market data
        market_data = await get_market_data()
        agent.load_market_data(market_data)
        
        simulation = agent.simulate_strategy(strategy.model_dump(), time_horizon)
        return simulation
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Error simulating strategy: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error simulating strategy: {str(e)}")

def calculate_portfolio_impact(strategy: Dict, current_positions: List) -> Dict:
    """Calculate the impact of the strategy on the user's portfolio"""
    logger.info("Calculating portfolio impact")
    
    # Extract portfolio metrics
    positions_value = 0
    pt_exposure = 0
    yt_exposure = 0
    
    for pos in current_positions:
        # Handle both TokenPosition objects and dicts
        if hasattr(pos, 'value_usd'):
            value = pos.value_usd
            token_type = pos.token_type
        else:
            value = pos.get("value_usd", 0)
            token_type = pos.get("token_type", "")
            
        positions_value += value
        
        if token_type == "PT":
            pt_exposure += value
        elif token_type == "YT":
            yt_exposure += value
    
    # Calculate current risk profile
    pt_percentage = (pt_exposure / positions_value) if positions_value > 0 else 0
    risk_profile = "low" if pt_percentage > 0.7 else "medium" if pt_percentage > 0.4 else "high"
    
    # Determine strategy risk level
    strategy_name = strategy.get("name", "")
    strategy_risk_level = {
        "Principal Protector": "low",
        "Balanced Approach": "medium",
        "Yield Maximizer": "medium-high",
        "Yield Speculation": "high"
    }.get(strategy_name, "medium")
    
    # Calculate expected impact
    expected_roi = strategy.get("expected_roi", 5.0)
    
    impact = {
        "before": {
            "total_value": positions_value,
            "risk_profile": risk_profile,
            "pt_percentage": round(pt_percentage * 100, 2),
            "yt_percentage": round(100 - (pt_percentage * 100), 2)
        },
        "after": {
            "total_value": positions_value * (1 + expected_roi/100),
            "risk_profile": strategy_risk_level,
            "pt_percentage": adjust_exposure_based_on_strategy(pt_percentage, strategy),
            "yt_percentage": 100 - adjust_exposure_based_on_strategy(pt_percentage, strategy)
        },
        "expected_changes": {
            "value_change_pct": expected_roi,
            "risk_change": compare_risk_profiles(risk_profile, strategy_risk_level),
            "liquidity_impact": determine_liquidity_impact(strategy)
        }
    }
    
    return impact

def adjust_exposure_based_on_strategy(current_pt_percentage, strategy):
    """Adjust PT exposure percentage based on strategy actions"""
    pt_percentage = current_pt_percentage * 100  # Convert to percentage
    
    for action in strategy.get("actions", []):
        token = action.get("token", "")
        percentage = action.get("percentage", 0)
        action_type = action.get("action", "")
        
        if token.startswith("PT") and action_type == "sell":
            # Selling PT reduces PT exposure
            reduction = (percentage / 100) * pt_percentage
            pt_percentage -= reduction
        elif token.startswith("PT") and action_type == "buy":
            # Buying PT increases PT exposure
            increase = (percentage / 100) * (100 - pt_percentage)
            pt_percentage += increase
        elif token.startswith("YT") and action_type == "sell":
            # Selling YT indirectly increases PT percentage
            increase = (percentage / 100) * (100 - pt_percentage) * 0.5  # Assume half goes to PT
            pt_percentage += increase
        elif token.startswith("YT") and action_type == "buy":
            # Buying YT reduces PT exposure
            reduction = (percentage / 100) * pt_percentage * 0.5  # Assume half comes from PT
            pt_percentage -= reduction
    
    return round(max(0, min(pt_percentage, 100)), 2)  # Ensure it's between 0-100%

def compare_risk_profiles(before, after):
    """Compare risk profiles and return a description of the change"""
    risk_levels = {"low": 1, "medium": 2, "medium-high": 3, "high": 4}
    before_level = risk_levels.get(before, 2)
    after_level = risk_levels.get(after, 2)
    
    if before_level == after_level:
        return "unchanged"
    elif before_level > after_level:
        return "decreased" if before_level - after_level > 1 else "slightly decreased"
    else:
        return "increased" if after_level - before_level > 1 else "slightly increased"

def determine_liquidity_impact(strategy):
    """Determine the liquidity impact of a strategy"""
    strategy_name = strategy.get("name", "")
    
    liquidity_impacts = {
        "Principal Protector": "increased",
        "Balanced Approach": "minimal",
        "Yield Maximizer": "moderate decrease",
        "Yield Speculation": "significant decrease"
    }
    
    return liquidity_impacts.get(strategy_name, "minimal")

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

# Root endpoint
@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Root endpoint with API information"""
    return {
        "api": "PT/YT ROI Optimization API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/optimize", "method": "POST", "description": "Optimize portfolio strategy"},
            {"path": "/explain", "method": "POST", "description": "Explain a strategy"},
            {"path": "/simulate", "method": "POST", "description": "Simulate a strategy"},
            {"path": "/health", "method": "GET", "description": "Health check"}
        ]
    }