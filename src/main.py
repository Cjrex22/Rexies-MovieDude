import os
import sqlite3
try:
    from scripts.login import login
    from modules import *
    from rich.console import Console
    console = Console(highlight=False)
    sep_line = "[cyan]-[/cyan]" * 50
except ImportError as error:
    print("\033[1;33m⚠️  Failed to import modules \033[0m", error)

def main(): 
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "database", "MovieDude.db")
    
    filter_watched = True
    filter_top_rank = False

    while True:
        user_id = login(db_path)

        while True:
            console.print(sep_line)
            console.print(
                "[bold]1[/bold] : Discover Movies Based on Your Activities (User Taste Clustering)\n"
                "[bold]2[/bold] : Find Similar Movies by Given Movie Title (Item Clustering & Latent Space)\n"
                "[bold]3[/bold] : Rate or Log a Movie\n"
                "[bold]4[/bold] : View My Watch & Rating History\n"
                "[bold]5[/bold] : Explore Movie Clusters & PCA Visualizer (Interactive Latent Map)\n"
                "[bold]6[/bold] : Switch User / Log Out\n"
                "[bold]7[/bold] : Exit"
            )
            
            choice = input("\nEnter your choice: ").strip()

            # 1. Discover Movies Based on Your Activities
            if choice == "1":
                console.print(sep_line)
                interests = find_user_interests(db_path, user_id)
                
                # Check for User Cold Start or No Positive Ratings (>= 7.5)
                if not interests or not interests.get("watched_movie_ids"):
                    console.print("\n[yellow]⚠️  You haven't rated or logged any movies yet![/yellow]")
                    console.print("[dim]💡 Displaying our Top-Rated Global Acclaimed Starter Picks to get you started.[/dim]")
                    console.print("[dim]Tip: Use Option 3 ('Rate or Log a Movie') to add movies you love (★ >= 7.5), and your feed will immediately personalize.[/dim]\n")
                    with console.status("Working..."):
                        recommend_movies = movie_recommender(db_path, user_id, {}, filter_watched, filter_top_rank)
                        print_titles(recommend_movies)
                elif not interests.get("positive_movie_ids"):
                    console.print("\n[yellow]⚠️  None of your logged movies have a rating of 7.5 or higher.[/yellow]")
                    console.print("[dim]💡 Personalized recommendations only consider movies rated ★ >= 7.5 to build your taste profile (movies rated below 7.5 are excluded).[/dim]")
                    console.print("[dim]Tip: Rate at least one movie ★ 7.5 or higher via Option 3 to generate personalized recommendations.[/dim]\n")
                    with console.status("Working..."):
                        recommend_movies = movie_recommender(db_path, user_id, interests, filter_watched, filter_top_rank)
                        print_titles(recommend_movies)
                else:
                    with console.status("Generating personalized feed via taste clustering..."):
                        recommend_movies = movie_recommender(db_path, user_id, interests, filter_watched=True, filter_top_rank=filter_top_rank)
                        if recommend_movies and "cluster_id" in recommend_movies[0]:
                            is_limited = recommend_movies[0].get("is_limited_history", False)
                            n_pos = len(interests.get("positive_movie_ids", []))
                            cohort = recommend_movies[0].get("user_cohort_name", "Taste Cohort")
                            if is_limited:
                                console.print(f"🌱 [bold cyan]Emerging Taste Profile[/bold cyan] [dim](Based on {n_pos} positively rated movie(s))[/dim]")
                                console.print(f"   • [dim]Preliminary Cohort:[/dim] [bold white]{cohort}[/bold white]")
                                console.print(f"   • [dim]Note: Both of your demonstrated preference areas are represented below.[/dim]")
                                console.print(f"   • [dim]Tip: As you rate more movies, recommendations will sharpen further.[/dim]\n")
                            else:
                                console.print(f"🎯 [bold cyan]Personalized Feed[/bold cyan] [dim](Based on {n_pos} positively rated movies)[/dim]")
                                console.print(f"   • [dim]Assigned User Cohort:[/dim] [bold white]{cohort}[/bold white]\n")
                        print_titles(recommend_movies)

            # 2. Find Similar Movies by Given Movie Title
            elif choice == "2":
                console.print(sep_line)
                while True:
                    title_query = input("Enter a movie title to find similar recommendations (or 'b' to go back): ").strip()
                    if title_query.lower() == 'b':
                        break
                    if recommendation_input := find_by_title(db_path, title_query):
                        from modules.clustering import get_clustering_model
                        cluster_model = get_clustering_model(db_path)
                        c_info = cluster_model.get_movie_cluster(recommendation_input["movie_id"])
                        if c_info:
                            console.print(f"📍 [dim cyan]K-Means Cluster #{c_info['cluster_id']}:[/dim cyan] [bold white]{c_info['cluster_name']}[/bold white] [dim](PCA: {c_info['pca_x']:.2f}, {c_info['pca_y']:.2f})[/dim]\n")
                        with console.status("Working..."):
                            recommend_movies = movie_recommender(db_path, user_id, recommendation_input, filter_watched, filter_top_rank)
                            print_titles(recommend_movies)
                        break

            # 3. Rate or Log a Movie
            elif choice == "3":
                console.print(sep_line)
                console.print("[bold cyan]Rate or Log a Movie to Your Profile[/bold cyan]\n")
                while True:
                    title_query = input("Enter movie title to rate (or 'b' to go back): ").strip()
                    if title_query.lower() == 'b':
                        break
                    movie_data = find_by_title(db_path, title_query)
                    if not movie_data:
                        continue

                    movie_id = movie_data["movie_id"]
                    movie_title = movie_data["title"]
                    year_str = f" ({movie_data['year']})" if movie_data.get('year') else ""

                    while True:
                        rate_input = input(f"\nEnter your rating for '{movie_title}{year_str}' (1.0 - 10.0, or press Enter for 8.0): ").strip()
                        if not rate_input:
                            user_rate = 8.0
                            break
                        try:
                            user_rate = float(rate_input)
                            if 1.0 <= user_rate <= 10.0:
                                break
                            print("⚠️ Please enter a rating between 1.0 and 10.0.")
                        except ValueError:
                            print("⚠️ Invalid number. Try again.")

                    liked_input = input(f"Did you like '{movie_title}'? (y/n, default 'y'): ").strip().lower()
                    liked = 0 if liked_input in ('n', 'no') else 1

                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                        INSERT INTO Users_data (user_id, movie_id, user_rate, liked)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(user_id, movie_id) DO UPDATE SET user_rate = excluded.user_rate, liked = excluded.liked
                        """, (user_id, movie_id, user_rate, liked))
                        conn.commit()

                    invalidate_ratings_cache()
                    console.print(f"\n[green]✅ Saved![/green] Added [bold white]{movie_title}{year_str}[/bold white] with rating [yellow]★ {user_rate:.1f}/10[/yellow] to your profile.\n")
                    break

            # 4. View My Watch & Rating History
            elif choice == "4":
                console.print(sep_line)
                console.print("[bold cyan]Your Watched & Rated Movies[/bold cyan]\n")
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                    SELECT ms.title, ms.release_year, ms.genres, ud.user_rate, ud.liked
                    FROM Users_data ud
                    JOIN Movies_sorted ms ON ud.movie_id = ms.movie_id
                    WHERE ud.user_id = ?
                    ORDER BY ud.user_rate DESC
                    """, (user_id,))
                    rows = cursor.fetchall()

                if not rows:
                    console.print("[yellow]You haven't rated or logged any movies yet![/yellow]")
                    console.print("💡 Use [bold]Option 3[/bold] to rate movies you've watched.\n")
                else:
                    console.print(f"Total Movies in Your Profile: [bold green]{len(rows)}[/bold green]\n")
                    for idx, (m_title, m_year, m_genres, rate, liked) in enumerate(rows, 1):
                        year_str = f" ({m_year})" if m_year else ""
                        liked_str = " ❤️" if liked else ""
                        tag = "[green][Seed: Rated ≥ 7.5][/green]" if rate >= 7.5 else "[dim yellow][Excluded: Rated < 7.5][/dim yellow]"
                        console.print(f"{idx}. [bold white]{m_title}[/bold white]{year_str} — [yellow]★ {rate:.1f}/10[/yellow]{liked_str} {tag}\n   [dim]{m_genres}[/dim]\n")
                    print()

            # 5. Explore Movie Clusters & PCA Visualizer
            elif choice == "5":
                console.print(sep_line)
                console.print("[bold cyan]Unsupervised Machine Learning: K-Means Clustering & PCA Map[/bold cyan]\n")
                
                with console.status("Computing latent clusters and PCA projections..."):
                    from modules.clustering import get_clustering_model
                    model = get_clustering_model(db_path)
                    metrics = model.get_cluster_metrics()
                    report_path = os.path.join(base_dir, "reports", "movie_clusters_pca.html")
                    model.generate_html_visualization(report_path)

                console.print(f"📊 [bold white]Unsupervised Model Architecture:[/bold white]")
                console.print(f"   • Item Clustering: [bold green]K={metrics['n_item_clusters']}[/bold green] (Inertia: [bold yellow]{metrics['item_inertia']:.1f}[/bold yellow], Silhouette: [bold yellow]{metrics['item_silhouette_score']:.4f}[/bold yellow])")
                console.print(f"   • User Clustering: [bold green]K={metrics['n_user_clusters']} Cohorts[/bold green] across {metrics['total_users']} users (Silhouette: [bold yellow]{metrics['user_silhouette_score']:.4f}[/bold yellow])")
                evr = metrics['pca_explained_variance_ratio']
                console.print(f"   • Movie 2D PCA Variance: [bold cyan]PC1: {evr[0]*100:.1f}%, PC2: {evr[1]*100:.1f}% (Total: {sum(evr)*100:.1f}%)[/bold cyan]")
                evr_u = metrics.get('user_pca_explained_variance_ratio', [])
                if evr_u:
                    console.print(f"   • User 2D PCA Variance:  [bold cyan]PC1: {evr_u[0]*100:.1f}%, PC2: {evr_u[1]*100:.1f}% (Total: {sum(evr_u)*100:.1f}%)[/bold cyan]\n")
                else:
                    print()

                console.print("[bold white]Discovered Taste Clusters:[/bold white]")
                for cid in range(metrics['n_clusters']):
                    c_name = model.cluster_names[cid]
                    c_count = len(model.df[model.df['cluster'] == cid])
                    sample_movies = model.df[model.df['cluster'] == cid].sort_values(by='rating_count', ascending=False)['title'].head(3).tolist()
                    console.print(f"  [bold cyan]#{cid}[/bold cyan] [bold white]{c_name}[/bold white] ({c_count} titles) — [dim]e.g. {', '.join(sample_movies)}[/dim]")

                console.print(f"\n[green]✅ Interactive PCA Visualizer ready:[/green]")
                console.print(f"   [dim]{report_path}[/dim]")
                
                open_choice = input("\nOpen interactive 2D cluster map in web browser now? (y/n, default 'y'): ").strip().lower()
                if open_choice in ('', 'y', 'yes'):
                    try:
                        import subprocess
                        subprocess.run(["open", report_path], check=False)
                        console.print("[dim]🚀 Opened in your default browser.[/dim]\n")
                    except Exception as e:
                        console.print(f"[dim yellow]Could not auto-open: {e}[/dim yellow]\n")

            # 6. Switch User / Log Out
            elif choice == "6":
                console.print("\n[yellow]Logging out...[/yellow]\n")
                break

            # 7. Exit
            elif choice == "7":
                console.print("\n[green]Goodbye![/green]\n")
                return

            else:
                console.print("[yellow]Invalid option![/yellow] Try again...")
        
def print_titles(recommend_titles):
    import re
    console.print("\n[bold cyan]Top 10 Recommended Movies:[/bold cyan]\n")

    for i, item in enumerate(recommend_titles, 1):
        if isinstance(item, dict):
            title = item.get("title", "")
            year = f" ({item['year']})" if item.get("year") else ""
            genres = f" • [dim]{item['genres']}[/dim]" if item.get("genres") else ""
            final_sc = item.get("final_score") or item.get("score")
            csim = item.get("content_similarity") or item.get("cosine_similarity")
            pscore = item.get("peer_support")
            rationale = item.get("rationale", "")

            console.print(f"  {i}. [bold white]{title}[/bold white]{year}{genres}")
            if final_sc is not None and csim is not None and pscore is not None:
                console.print(f"     ↳ [bold yellow]Final Rank Score: {final_sc:.3f}[/bold yellow] [dim cyan](Content Sim: {csim:.3f}, Peer Cohort Support: {pscore:.3f})[/dim cyan]")
            elif csim is not None:
                console.print(f"     ↳ [bold yellow]Cosine Similarity: {csim:.3f}[/bold yellow]")

            if rationale:
                clean_rat = re.sub(r'^Rank Score:\s*[\d\.]+\s*\([^)]*\)\s*•\s*', '', rationale)
                clean_rat = re.sub(r'^Cosine Similarity:\s*[\d\.]+\s*•\s*', '', clean_rat)
                console.print(f"     ↳ [dim]Match Rationale: {clean_rat}[/dim]")
            print()
        else:
            print(f"  {i}. {item}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting MovieDude. Goodbye!")
