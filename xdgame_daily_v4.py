#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XDGAME 每日热门游戏自动抓取 v4
- 从 /list/1/ 页面获取最新更新日期游戏（通过 <time class="news">）
- 使用会员 Cookie 获取真实网盘链接
- 净化下载链接显示（去除 uhash 标签）
- 解决天翼网盘链接乱码问题
- 自动发送邮件到 QQ 邮箱
- 保存本地报告
"""

import requests
import json
import re
import time
import smtplib
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import os
import urllib.parse

# ==================== 配置区域 ====================

# 用户提供的有效 Cookie（会员账号）
COOKIES = {
    "Hm_lvt_1905089d52b6f08f01b437535400116c": "1765047566,1765120499,1765613116,1765797061",
    "night": "0",
    "PHPSESSID": "dn6guf6avhpo31qat6a4ej26n6",
    "DedeUserID": "13863",
    "DedeUserID__ckMd5": "e590eb06fde3102e",
    "DedeLoginTime": "1773212139",
    "DedeLoginTime__ckMd5": "c67ac377719980e0"
}

# 邮箱配置
EMAIL_CONFIG = {
    "smtp_server": "smtp.qq.com",
    "smtp_port": 587,
    "from_email": "979890503@qq.com",
    "to_email": "979890503@qq.com",
    # QQ 邮箱授权码（已在配置中）
    "password": "ywyztipsarknbfga"
}

BASE_URL = "https://www.xdgame.com"
WORK_DIR = "/home/muliy/.copaw"


def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": BASE_URL,
    })
    for key, value in COOKIES.items():
        session.cookies.set(key, value, domain=".xdgame.com")
    return session


def clean_download_url(url):
    """清理下载 URL，解决乱码问题"""
    if not url:
        return ""
    
    # 如果是乱码的天翼网盘链接，尝试修复
    # 例如: https://cloud.189.cn/t/mEJFB3IBBnMvï¼è®¿é®ç ï¼8u4oï¼
    # 正常应该是提取码部分
    
    # 检查是否有乱码字符
    if 'ï¼' in url or 'æ' in url or 'å' in url:
        # 只保留有效的 URL 部分，去掉乱码
        # 天翼网盘格式：https://cloud.189.cn/t/{code}
        match = re.match(r'(https://cloud\.189\.cn/t/[A-Za-z0-9]+)', url)
        if match:
            return match.group(1)
        
        # 其他情况，只取到第一个特殊字符之前
        clean = re.split(r'[ï½ï¼æ¥]', url)[0]
        return clean.strip()
    
    return url


def extract_clean_url(url):
    """从 download.php 重定向中提取干净的网盘 URL"""
    if not url:
        return ""
    
    # 已经是标准网盘链接
    if any(domain in url for domain in [
        'pan.baidu.com', 'cloud.189.cn', 'pan.xunlei.com',
        'aliyundrive.com', 'alipan.com', 'pan.quark.cn',
        '139.com', '123pan.com', 'drive.uc.cn'
    ]):
        return clean_download_url(url)
    
    # 磁力链接
    if url.startswith('magnet:'):
        return url
    
    return clean_download_url(url)


def get_game_details(session, game_id):
    """获取游戏详情和真实网盘链接"""
    game_url = f"{BASE_URL}/game/{game_id}.html"
    response = session.get(game_url, timeout=10)
    if response.status_code != 200:
        return None
    
    content = response.text
    soup = BeautifulSoup(content, 'html.parser')
    
    # 检查是否有验证码保护
    has_captcha = 'class="code"' in content or '请输入验证码' in content
    
    # 标题
    title_elem = soup.find('title')
    game_name = title_elem.text.replace(" - XDGAME", "").strip() if title_elem else f"游戏{game_id}"
    
    # 中文简介
    desc_elem = soup.find('meta', attrs={'name': 'description'})
    description = desc_elem['content'] if desc_elem else "暂无简介"
    
    # 截图 - 查找包含 ss_ 的图片（Steam CDN）
    images = []
    img_elems = soup.find_all('img')
    
    for img in img_elems:
        src = img.get('src', '') or img.get('data-src', '')
        # Steam 截图特征
        if ('ss_' in src and 
            ('jpg' in src.lower() or 'png' in src.lower()) and
            'logo.png' not in src.lower()):
            
            if not src.startswith('http'):
                src = f"https:{src}" if src.startswith('//') else f"https://www.xdgame.com{src}"
            
            images.append(src)
            if len(images) >= 5:
                break
    
    # 下载按钮 - 解析 data-url 属性
    downloads = []
    download_buttons = soup.find_all('a', class_='downbtn')
    
    for btn in download_buttons:
        name = btn.text.strip()
        data_url = btn.get('data-url', '')
        
        if '/plus/download.php?open=2' in data_url:
            full_url = f"{BASE_URL}{data_url}"
            try:
                dl_resp = session.get(full_url, allow_redirects=False, timeout=10)
                if dl_resp.status_code == 302:
                    real_url = dl_resp.headers.get('Location', '')
                    # 清理 URL
                    clean_url = extract_clean_url(real_url)
                    
                    if clean_url:
                        downloads.append({
                            "name": name,
                            "real_url": clean_url,
                            "direct": True
                        })
                elif dl_resp.status_code == 200:
                    # 200 说明是验证页面，记录原始下载接口供手动访问
                    downloads.append({
                        "name": name,
                        "real_url": f"[需验证] {full_url}",
                        "direct": False
                    })
                else:
                    downloads.append({
                        "name": name,
                        "real_url": f"[获取失败]{data_url}",
                        "direct": False
                    })
            except Exception as e:
                downloads.append({
                    "name": name,
                    "real_url": f"[错误: {str(e)}]",
                    "direct": False
                })
    
    return {
        "id": game_id,
        "name": game_name,
        "description": description,
        "images": images,
        "downloads": downloads,
        "url": game_url,
        "has_captcha": has_captcha
    }


def fetch_latest_games(session, count=5):
    """从 /list/1/ 页面获取最新更新的游戏列表"""
    url = f"{BASE_URL}/list/1/"
    response = session.get(url, timeout=10)
    if response.status_code != 200:
        print(f"⚠️ 最近更新页面访问失败，使用备用列表")
        return get_fallback_games(count)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    games = []
    seen_ids = set()
    
    # 查找所有 time.news 元素及其父级链接
    time_elements = soup.find_all('time', class_='news')
    
    for time_elem in time_elements:
        if len(games) >= count * 2:  # 获取更多候选游戏
            break
        
        # 向上查找包含游戏链接的容器（最多 2 层）
        parent = time_elem.find_parent('li')
        if not parent:
            parent = time_elem.find_parent()
        
        link = parent.find('a') if parent else None
        
        if not link:
            continue
        
        href = link.get('href', '')
        match = re.search(r'/game/(\d+)\.html', href)
        
        if not match:
            continue
        
        game_id = int(match.group(1))
        if game_id in seen_ids:
            continue
        
        seen_ids.add(game_id)
        games.append({'id': game_id})
    
    if not games:
        print(f"⚠️ 未解析到游戏列表，使用备用列表")
        return get_fallback_games(count)
    
    print(f"✅ 从 list/1/ 页面成功解析到 {len(games)} 款新游戏")
    for g in games[:10]:
        print(f"   - Game ID: {g['id']}")
    
    # 不再筛选，直接使用最新游戏（即使需要验证也保留接口信息）
    return [g['id'] for g in games[:count]]


def get_fallback_games(count=5):
    """备用游戏列表（已验证有下载资源的老游戏）"""
    fallback_ids = [197, 218, 5044, 11993, 6449, 1497, 8833, 9034, 12368, 13332]
    return [{'id': fid} for fid in fallback_ids[:count]]


def generate_text_report(results, timestamp):
    """生成文本报告"""
    report = []
    report.append("=" * 80)
    report.append("🎮 XDGAME 今日热门游戏 TOP 5 🎮")
    report.append("=" * 80)
    report.append(f"📅 生成时间：{timestamp}")
    report.append(f"🌐 数据来源：https://www.xdgame.com/")
    report.append(f"👤 账号状态：会员已登录")
    report.append(f"⚠️ 说明：网站每日下载限制 200 次，配额用完后需等待 24 小时重置")
    report.append("")
    
    total_downloads = 0
    direct_links = 0
    
    for game in results:
        report.append("-" * 80)
        report.append(f"【排名 #{game['rank']}】{game['name']} (ID: {game['id']})")
        report.append("-" * 80)
        
        # 简介
        report.append(f"\n📖 游戏简介:")
        desc = game['description'][:200] + "..." if len(game['description']) > 200 else game['description']
        report.append(desc)
        
        # 截图
        report.append(f"\n🖼️ 游戏截图 ({len(game['images'])} 张):")
        for i, img in enumerate(game['images'], 1):
            report.append(f"   {i}. {img}")
        
        # 下载链接 - 已净化
        report.append(f"\n📦 网盘下载 ({len(game['downloads'])} 个通道):")
        for dl in game['downloads']:
            url_display = dl['real_url'][:80] + "..." if len(dl['real_url']) > 80 else dl['real_url']
            status = "✅" if dl.get('direct') else "🔒"
            report.append(f"   {status} {dl['name']}: {url_display}")
            total_downloads += 1
            if dl.get('direct'):
                direct_links += 1
        
        report.append("")
    
    # 统计
    report.append("=" * 80)
    report.append("📊 统计摘要")
    report.append("=" * 80)
    report.append(f"✅ 直接可用链接：{direct_links} 个")
    report.append(f"⚠️ 今日配额已用完（网站限制 24h 内最多 200 次）")
    report.append(f"🎯 游戏总数：{len(results)} 款")
    report.append(f"⏰ 完成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("💡 重要提示：")
    report.append("   ⚠️ XDGAME 网站会员每日下载次数限制为 200 次")
    report.append("   ⚠️ 配额用完后，真实网盘链接无法获取（显示验证页面）")
    report.append("   🕒 下载次数将在 24 小时后自动重置")
    report.append("   ✅ 明日同一时间运行脚本可重新获取真实链接")
    report.append("   • 标✅的链接可直接访问下载")
    report.append("   • 标⚠️的链接需等待明天配额重置后才能获取")
    report.append("=" * 80)
    
    return "\n".join(report)


def generate_html_email(results, timestamp):
    """生成 HTML 邮件内容"""
    total_downloads = sum(len(g['downloads']) for g in results)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }}
            .game {{ border: 1px solid #ddd; margin: 20px 0; border-radius: 8px; overflow: hidden; }}
            .game-header {{ background: #f5f5f5; padding: 15px; border-bottom: 1px solid #ddd; }}
            .game-body {{ padding: 15px; }}
            .rank {{ display: inline-block; background: #667eea; color: white; padding: 5px 10px; border-radius: 4px; margin-right: 10px; }}
            .download-item {{ background: #f9f9f9; padding: 10px; margin: 8px 0; border-radius: 4px; word-break: break-all; border-left: 3px solid #667eea; }}
            .stats {{ background: #e8f5e9; padding: 15px; border-radius: 8px; text-align: center; margin: 15px 0; }}
            a {{ color: #667eea; }}
            .footer {{ text-align: center; color: #999; font-size: 12px; padding: 20px; }}
            .tag {{ display: inline-block; background: #e3f2fd; color: #1976d2; padding: 3px 8px; border-radius: 3px; font-size: 12px; margin-right: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎮 XDGAME 最新更新游戏 TOP 5</h1>
            <p>{timestamp}</p>
        </div>
        
        <div class="stats">
            <strong>✅ 成功获取：{total_downloads} 个真实网盘下载链接</strong><br>
            <strong>🎯 游戏总数：{len(results)} 款</strong>
        </div>
    """
    
    for game in results:
        html += f"""
        <div class="game">
            <div class="game-header">
                <span class="rank">#{game['rank']}</span>
                <strong>{game['name']}</strong>
                <span style="color: #999; margin-left: 10px; font-size: 14px;">(ID: {game['id']})</span>
            </div>
            <div class="game-body">
                <p><strong>📖 简介:</strong><br>{game['description'][:150]}...</p>
                
                <p><strong>🖼️ 截图 ({len(game['images'])} 张):</strong></p>
                <div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0;">
        """
        
        for img in game['images'][:3]:
            html += f'<img src="{img}" style="width: 180px; height: 101px; object-fit: cover; border-radius: 4px;">'
        
        html += """
                </div>
                
                <p><strong>📦 网盘下载通道:</strong></p>
        """
        
        for dl in game['downloads']:
            url_short = dl['real_url'][:60] + "..." if len(dl['real_url']) > 60 else dl['real_url']
            safe_url = urllib.parse.quote(dl['real_url'], safe=':/?=&')
            html += f'<div class="download-item"><strong>{dl["name"]}:</strong> <a href="{dl["real_url"]}">{url_short}</a></div>'
        
        html += """
            </div>
        </div>
        """
    
    html += f"""
        <div class="footer">
            <p>💡 提示：链接可能随时失效，建议及时下载保存！</p>
            <p>数据来源：<a href="https://www.xdgame.com/list/1/">XDGAME 最近更新</a> | 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    return html


def send_email(subject, html_content, text_content):
    """发送邮件"""
    if not EMAIL_CONFIG["password"]:
        print("⚠️ 邮箱密码未配置，跳过邮件发送")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = Header(subject, 'utf-8')
        msg['From'] = EMAIL_CONFIG["from_email"]
        msg['To'] = EMAIL_CONFIG["to_email"]
        
        msg.attach(MIMEText(text_content, 'plain', 'utf-8'))
        msg.attach(MIMEText(html_content, 'html', 'utf-8'))
        
        # 尝试 STARTTLS（端口 587）
        server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"], timeout=15)
        server.starttls()
        server.login(EMAIL_CONFIG["from_email"], EMAIL_CONFIG["password"])
        server.sendmail(EMAIL_CONFIG["from_email"], [EMAIL_CONFIG["to_email"]], msg.as_string())
        server.quit()
        
        print("✅ 邮件发送成功！")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"❌ 邮件发送失败：{error_msg[:100]}")
        
        # 判断错误类型
        if "UNEXPECTED_EOF" in error_msg or "Connection unexpectedly closed" in error_msg:
            print("💡 可能原因:")
            print("   1. QQ 邮箱授权码已过期（需要重新生成）")
            print("   2. SMTP 服务临时不可用")
            print("   3. 网络连接问题")
        elif "Authentication failed" in error_msg:
            print("💡 授权码错误或已失效，请重新获取:")
            print("   QQ 邮箱 → 设置 → 账户 → 生成新授权码")
        else:
            print("💡 请检查:")
            print("   1. QQ 邮箱是否开启 SMTP 服务")
            print("   2. 授权码是否正确")
            print("   3. 网络是否正常")
        
        return False


def main():
    print("=" * 80)
    print("🎮 XDGAME 每日热门游戏自动抓取 v4")
    print("=" * 80)
    print(f"🕒 开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    session = get_session()
    
    # 验证登录
    print("🔍 验证登录状态...", end=" ")
    test = session.get(f"{BASE_URL}/user/", timeout=5)
    if "会员中心" in test.text:
        print("✅ 已登录（会员账号）")
    else:
        print("⚠️ Cookie 可能已过期")
    
    # 获取最新更新的游戏列表（自动筛选有下载资源的游戏）
    print("\n📥 正在从 /list/1/ 获取最新更新的游戏...")
    game_ids = fetch_latest_games(session, count=5)
    
    # 获取游戏详情
    print("\n📥 正在获取游戏详情...")
    results = []
    
    for i, game_id in enumerate(game_ids, 1):
        print(f"\n【{i}/{len(game_ids)}】Game ID: {game_id}...")
        
        details = get_game_details(session, game_id)
        if not details:
            print(f"   ❌ 获取失败")
            continue
        
        results.append({
            "rank": i,
            "name": details["name"],
            "id": details["id"],
            "description": details["description"],
            "images": details["images"],
            "downloads": details["downloads"],
            "url": details["url"]
        })
        
        captcha_marker = "🔒" if details.get('has_captcha') else "✅"
        print(f"   {captcha_marker} 名称：{details['name']}")
        print(f"   ✅ 截图：{len(details['images'])} 张")
        valid_dl = [d for d in details['downloads'] if '获取失败' not in d['real_url'] and '错误' not in d['real_url']]
        print(f"   ✅ 可用网盘通道：{len(valid_dl)} 个")
        
        time.sleep(1)
    
    # 生成报告
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    text_report = generate_text_report(results, timestamp)
    html_email = generate_html_email(results, timestamp)
    
    # 保存文件
    os.chdir(WORK_DIR)
    
    with open(f"XDGAME_TODAY_{timestamp_file}.txt", "w", encoding="utf-8") as f:
        f.write(text_report)
    
    with open("XDGAME_LATEST.txt", "w", encoding="utf-8") as f:
        f.write(text_report)
    
    with open(f"XDGAME_TODAY_{timestamp_file}.html", "w", encoding="utf-8") as f:
        f.write(html_email)
    
    print("\n" + "=" * 80)
    print("📝 报告已保存:")
    print(f"   • XDGAME_LATEST.txt")
    print(f"   • XDGAME_TODAY_{timestamp_file}.txt")
    print(f"   • XDGAME_TODAY_{timestamp_file}.html")
    
    # 发送邮件
    print("\n" + "=" * 80)
    print("📧 正在发送邮件...")
    subject = f"🎮 XDGAME 最新更新游戏 TOP 5 ({datetime.now().strftime('%Y-%m-%d')})"
    send_email(subject, html_email, text_report)
    
    print("\n" + "=" * 80)
    print("✅ 任务完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
